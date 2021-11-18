# -*- coding: utf-8 -*-"
"""
Created on 12/13/2020  1:38 PM


@author: Zhuo
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy import stats

from core.statistic_analysis import render_mpl_table


def pearson_corr(df, filepath, xrotation=0, col_name_list=None):
    """
    Pearson Correlation (Numeric data)

    The correlation coefficient has values between -1 to 1
    - A value closer to 0 implies weaker correlation (exact 0 implying no correlation)
    - A value closer to 1 implies stronger positive correlation
    - A value closer to -1 implies stronger negative correlation

    Parameters
    ----------
    df: dataframe,
        Data to do the pearson correlation.

    filepath: string,
        saved directory.

    xrotation: int,
        the rotation angle of the x labels.
    """
    sns.set(style='ticks', color_codes=True)

    plt.figure(dpi=200)
    cor = df.astype(float).corr()
    # mask = np.triu(np.ones_like(cor, dtype=np.bool))

    sns.heatmap(
        cor,
        linewidths=0.05,
        square=True,
        cmap="coolwarm",
        # mask=mask,
        linecolor='white',
        annot=False,
        # fmt=".1f",
        # annot_kws={'size': 10},
        cbar_kws={"shrink": .5},
    )

    if col_name_list:
        xticks_list = np.arange(0.5, len(col_name_list)+0.5, 1)
        plt.xticks(xticks_list, col_name_list, rotation=xrotation, fontsize=5)
        plt.yticks(xticks_list, col_name_list, fontsize=5)
    else:
        plt.xticks(rotation=xrotation, fontsize=5)
        plt.yticks(fontsize=5)

    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values

    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()


def univariate_analysis(
        data,
        label_str,
        tab_title,
        save=True,
        save_dir="",
        ignore_col_list=None,
        sort_col='Variable',
):
    """
    Get the ROC univariate analysis figure and table

    :param data: Dataframe
        data contains variables and label values

    :param label_str: String
        the label's name

    :param tab_title: String
        The name of the figure

    :param save: bool
        save the results or not

    :param save_dir: string
        The directory of the saved figure and table

    :param ignore_col_list: list,
        The ignored column list

    :param sort_col: string,
        sort the dataframe by the sort_col column.

    :return data_df: Dataframe
        Summary table

    :return sig05_cols: List,
        Significance variables that P-value <= 0.05

    :return sig10_cols: List,
        Significance variables that P-value <= 0.10
    """
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if ignore_col_list is None:
        ignore_col_list = []
    try:
        data = data.drop(columns=ignore_col_list)
    except:
        pass

    feature_list = data.columns.values.tolist()
    feature_list.remove(label_str)

    x = data[feature_list]
    y = data[label_str]

    row_list = []
    col_list = []
    for i, col in enumerate(feature_list):
        df_feature = sm.add_constant(x[col])
        model = sm.Logit(y.astype(int), df_feature.astype(float))
        result = model.fit(method='bfgs')
        # print(result.summary())

        y_test = y.astype(int)
        y_pred = result.predict()

        # plot roc
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
        auc_ = metrics.roc_auc_score(y_test, y_pred)
        col_list.append(col)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred.round()).ravel()

        row = []
        # cut-off point
        threshold = threshold[np.argmax(tpr - fpr)].round(2)
        cutoff = np.subtract(np.max(df_feature[col]), np.min(df_feature[col]), dtype=np.float32) * threshold + np.min(
            df_feature[col])
        row.append(np.round(cutoff, 1))

        # sensitivity
        row.append(np.round(tp / (tp + fn), 2))

        # specificity
        row.append(np.round(tn / (tn + fp), 2))

        # AUC
        row.append(auc_.round(2))

        # OR
        or_ = np.exp(result.params)[1]
        row.append(np.round(or_, 2))
        # row.append("{:0.3e}".format(or_))  # np.round(or_, 2))

        # Confidence interval
        conf = result.conf_int()
        row.append([np.round(np.exp(conf).iloc[1, 0], 2), np.round(np.exp(conf).iloc[1, 1], 2)])

        # p-value
        row.append(result.pvalues[1].round(3))

        row_list.append(row)

    data_df = pd.DataFrame(row_list, columns=['Cut-off', 'Sensitivity', 'Specificity', 'AUC',
                                              'OR', '95% CI', 'P_value'])
    data_df.insert(0, 'Variables', col_list)
    data_df.sort_values(by=sort_col, inplace=True)

    # save
    if save:
        data_df.to_csv(os.path.join(save_dir, 'Uni_{}.csv'.format(tab_title)), index=False)
        render_mpl_table(data_df, os.path.join(save_dir, 'Fig_uni_{}.png'.format(tab_title)))

    # significant variables
    sig05_cols = data_df[data_df['P_value'] <= 0.05]['Variables'].tolist()
    sig10_cols = data_df[data_df['P_value'] <= 0.10]['Variables'].tolist()

    # get the minimum significant p-value
    if data_df.shape[0] > 0:
        min_p = min(data_df['P_value'].tolist())
        max_auc = max(data_df['AUC'].tolist())
    else:
        min_p = np.nan
        max_auc = np.nan

    return data_df, sig05_cols, sig10_cols, min_p, max_auc


def multivariate_analysis(data, feature_list, feature_name_list, label_str, ignore_col_list, fig_title, save_dir,
                          multivariate_model='OLS', line_index_list=None, y_axis='aic',
                          rotation=0, horizontal_bar=False, y_min=None, y_max=None, line_h_list=None):
    """
    Compare the multivariate analysis from different combinations of the variables.


    Parameters
    ----------
    data: dataframe,
        data contains variables and label values

    feature_list: List
        the list of the combinations of the variables

    feature_name_list: List
        the list of the combinations' name

    label_str: String,
        The name of the label column.

    ignore_col_list: list,
        Ignored columns in multivariate analysis

    fig_title: String
        The name of the fig

    save_dir: string
        The directory of the saved figure and table

    multivariate_model: string,
        The model for the multivariate analysis.

    Return
    ------
    data_df: Dataframe
        Summary table

    """
    row_list = []
    y = data[label_str]
    file = open(os.path.join(save_dir, 'Log_multi_{}.txt'.format(fig_title)), 'w')  # save multi results.

    y_axis_list = []
    llf_list = []
    shape_list = []

    for i, cols in enumerate(feature_list):
        # drop columns that in the ignore_col_list
        cols = [item for item in cols if item not in ignore_col_list]

        # save the shape of each bar
        shape_list.append(data[cols].shape[1])

        # add constant in the data.
        df_feature = sm.add_constant(data[cols])

        if multivariate_model == 'logit':
            # logit regression model
            model = sm.Logit(y.astype(float), df_feature.astype(float))
            result = model.fit(method='bfgs')
        elif multivariate_model == 'OLS':
            # OLS model
            model = sm.OLS(y.astype(float), df_feature.astype(float))
            result = model.fit()
        else:
            raise ValueError('The multivariate_model should only be \'logit\', \'OLS\'. ')

        # print the results
        print(result.summary(), file=file)
        print("\nOR: \n", np.exp(result.params), file=file)
        print("conf: \n", np.exp(result.conf_int()), file=file)

        y_test = y.astype(int)
        y_pred = result.predict()

        # results
        fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
        auc_ = metrics.roc_auc_score(y_test, y_pred)

        # incremental association
        if y_axis == 'acc':
            # accuracy
            acc = metrics.accuracy_score(y_test, y_pred.round())
            y_axis_list.append(acc)
        elif y_axis == 'auc':
            # AUC
            auc_ = auc_.round(2)
            y_axis_list.append(auc_)
        elif y_axis == 'aic':
            # AIC
            aic = result.aic.round(2)
            y_axis_list.append(aic)
        else:
            raise ValueError('The output could only be \'acc\', \'auc\' and \'aic\'.')

        # log likelihood
        llf = result.llf
        llf_list.append(llf)

        # plot multivariate analysis AUC figure
        label = feature_name_list[i] + ', auc=' + str(np.round(auc_, 2))
        plt.plot(fpr, tpr, label=label)

        # Calculate the results table of multivariate analysis.
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred.round()).ravel()
        row = [threshold[np.argmax(tpr - fpr)].round(2),  # cut-off point
               np.round(tp / (tp + fn), 2),  # sensitivity
               np.round(tn / (tn + fp), 2),  # specificity
               auc_.round(2),  # AUC
               result.aic.round(2),  # AIC
               result.rsquared,  # R-squared
               ]

        row_list.append(row)

    plt.legend(loc=4)
    plt.savefig(os.path.join(save_dir, 'Fig_multi_AUC_{}.png'.format(fig_title)),
                dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    # plot incremental_association
    if not y_min:
        y_max = max(y_axis_list) + 0.5
        y_min = min(y_axis_list) - 0.5

    # plot bar figure
    feature_name_list_bar = feature_name_list.copy()
    for i in range(len(feature_name_list_bar)):
        if i % 2 == 0:
            feature_name_list_bar[i] = "\n" + feature_name_list_bar[i]
        else:
            feature_name_list_bar[i] = "\n\n\n" + feature_name_list_bar[i]

    x = range(len(feature_name_list_bar))
    if horizontal_bar:
        plt.barh(x, y_axis_list, color='b', alpha=0.6)
        plt.yticks(x, feature_name_list_bar)
        plt.xlim(y_min, y_max)
        for i, v in enumerate(y_axis_list):
            plt.text(v, i, '{:.02f}'.format(v), color='b', ha='left', va='center')
    else:
        plt.bar(x, y_axis_list, color='b', alpha=0.6, width=0.5)
        plt.xticks(x, feature_name_list_bar, rotation=rotation)
        plt.ylim(y_min, y_max)
        for i, v in enumerate(y_axis_list):
            plt.text(i, v, '{:.02f}'.format(v), color='b', ha='center', va='bottom')

    # the step of the red line
    if line_index_list:
        h_step = (y_axis_list[0] - y_min) / len(line_index_list)

        for i, v in enumerate(line_index_list):
            dof = np.abs(shape_list[v[0]] - shape_list[v[1]])
            lr, p = lrtest(min(llf_list[v[0]], llf_list[v[1]]), max(llf_list[v[0]], llf_list[v[1]]), dof=dof)

            # the location of the red line
            if line_h_list:
                h = line_h_list[i]
            else:
                h = y_axis_list[0] - h_step * i - 0.05

            plt.plot(v, [h, h], color='r')
            plt.text(np.mean(v), h, 'lr={:.02f}, p={:.03f}'.format(lr, p), color='r', ha='center', va='bottom')

    plt.grid(axis='y')
    plt.savefig(os.path.join(save_dir, 'Fig_incremental_{}.png'.format(fig_title)), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    # save tables
    data_df = pd.DataFrame(row_list, columns=['Cut-off', 'Sensitivity', 'Specificity', 'AUC', "AIC", "R-Squared"])
    data_df.insert(0, ' ', feature_name_list)

    data_df.to_csv(os.path.join(save_dir, 'Multi_AUC_{}.csv'.format(fig_title)), index=False)
    render_mpl_table(data_df, os.path.join(save_dir, 'Fig_table_multi_{}.png'.format(fig_title)))
    return data_df


def lrtest(llmin, llmax, dof):
    """Likelihood ratio test
    Parameters
    ----------
    llmin: float
        the min of the log likelihood
    llmax: float
        the max of the log likelihood
    dof: int
        degree of freedom

    Return
    ------
    lr: float
        likelihood ratio
    p:  float
        p-value
    """
    lr = 2 * (llmax - llmin)
    p = stats.chi2.sf(lr, dof)  # llmax has dof more than llmin
    return lr, p


def incremental_association(data, feature_list, feature_name_list, label_str, line_index_list, ignore_col_list, fig_title,
                            save_dir, y_axis='auc', line_start_from_lowest=True):
    """

    Parameters 
    ----------
    data : dataframe,
        Inputted dataframe

    feature_list: List,
        The list of the combinations of the variables

    feature_name_list: List
        The list of the combinations' name

    label_str: String,
        The name of the label column.

    line_index_list: List,
        The list of the likelihood ratio line.

    ignore_col_list: list,
        Ignored columns in multivariate analysis

    fig_title: String
        The name of the fig

    save_dir: string
        The directory of the saved figure and table

    y_axis: string,
        The y axis of the incremental association figure. Can only be 'auc', 'aic', 'acc'.

    line_start_from_lowest: bool,
        If the likelihood ratio line starts from the lowest bar (True) or starts from the first bar (False).
    """

    global h
    y_test = data[label_str]

    acc_list = []
    llf_list = []
    shape_list = []

    for f_list in feature_list:
        shape_list.append(data[f_list].shape[1])
        df_f_list = sm.add_constant(data[f_list])

        model = sm.Logit(y_test, df_f_list.astype(float))
        result = model.fit(method='bfgs')
        y_pred = result.predict()

        if y_axis == 'acc':
            # accuracy
            acc = metrics.accuracy_score(y_test, y_pred.round())
            acc_list.append(acc)
        elif y_axis == 'auc':
            # AUC
            auc_ = metrics.roc_auc_score(y_test, y_pred).round(2)
            acc_list.append(auc_)
        elif y_axis == 'aic':
            # AIC
            aic = result.aic()
            acc_list.append(aic)
        else:
            raise ValueError('The output could only be \'acc\', \'auc\' and \'aic\'.')
        # log likelihood
        llf = result.llf
        llf_list.append(llf)

    sns.set(font_scale=1, style='white', color_codes=True)

    y_max = max(acc_list) + 0.1
    y_min = min(acc_list) - 0.3

    # plot bar figure
    x = range(len(feature_name_list))
    plt.bar(x, acc_list, color='b', alpha=0.6, width=0.5)
    plt.xticks(x, feature_name_list, rotation=60)
    plt.ylim(y_min, y_max)
    for i, v in enumerate(acc_list):
        plt.text(i, v, '{:.02f}'.format(v), color='b', ha='center', va='bottom')

    # the step of the red line
    h_step = (acc_list[0] - y_min) / len(line_index_list)

    for i, v in enumerate(line_index_list):
        dof = np.abs(shape_list[v[0]] - shape_list[v[1]])
        lr, p = lrtest(min(llf_list[v[0]], llf_list[v[1]]), max(llf_list[v[0]], llf_list[v[1]]), dof=dof)

        # the location of the red line
        if line_start_from_lowest:
            h = acc_list[0] - h_step * i - 0.05
        else:
            if i == 0:
                h = acc_list[v[0]] - h_step * i - 0.07
                h_step = (acc_list[v[0]] - y_min) / len(line_index_list)
            else:
                h = h - h_step

        plt.plot(v, [h, h], color='r')
        plt.text(np.mean(v), h, 'lr={:.02f}, p={:.03f}'.format(lr, p), color='r', ha='center', va='bottom')

    plt.grid(axis='y')
    plt.savefig(os.path.join(save_dir, 'Fig_Incremental.png'), dpi=200, bbox_inches="tight")
    plt.show()
