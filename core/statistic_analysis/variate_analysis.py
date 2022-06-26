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

import statsmodels
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy import stats

from core.statistic_analysis import render_mpl_table
from core.statistic_analysis.DeLong import delong_roc_variance, delong_roc_test


# def review_pearson_filter()
def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(stats.pearsonr(df[r], df[c])[1], 4)
    return pvalues


def pearson_corr(df, filepath, col_name_list=None, filter_corr=None,
                 xrotation=0, xticks_fontsize=5, xticks_ha='center',
                 annot=False, annot_kws={'size': 10},
                 part_cor=False):
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
    cor = cor.dropna(0, "all")
    cor = cor.dropna(1, "all")
    # mask = np.triu(np.ones_like(cor, dtype=np.bool))
    if part_cor:
        cor = cor.iloc[:21, :21]
    sns.heatmap(
        cor,
        linewidths=0.05,
        square=True,
        cmap="coolwarm",
        # mask=mask,
        linecolor='white',
        annot=annot,
        fmt=".1f",
        annot_kws=annot_kws,
        cbar_kws={"shrink": .5},
    )

    cor.to_csv(filepath.replace('.png', '.csv'))
    p_df = calculate_pvalues(df)
    p_df.to_csv(filepath.replace('.png', '_p.csv'))

    col_name_list = cor.columns.values.tolist()
    col_name_list = [sub.replace('original_', '') for sub in col_name_list]

    xticks_list = np.arange(0.5, len(col_name_list) + 0.5, 1)
    plt.xticks(xticks_list, col_name_list, rotation=xrotation, fontsize=xticks_fontsize, ha=xticks_ha)
    plt.yticks(xticks_list, col_name_list, fontsize=xticks_fontsize)

    # fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values

    plt.savefig(filepath, dpi=400, bbox_inches="tight")
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
        method='logit',
        fit_method='bfgs',
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
        if method == 'logit':
            model = sm.Logit(y.astype(int), df_feature.astype(float))
        elif method == 'ols':
            model = sm.OLS(y.astype(int), df_feature.astype(float))
        elif method == 'gls':
            model = statsmodels.regression.linear_model.GLS(y.astype(int), df_feature.astype(float))
        else:
            raise ValueError('Unsupported methods: ', method)
        result = model.fit(method=fit_method)
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
        try:
            or_ = np.exp(result.params)[1]
        except:
            or_ = np.exp(result.params)
        row.append(np.round(or_, 2))
        # row.append("{:0.3e}".format(or_))  # np.round(or_, 2))

        # Confidence interval
        conf = result.conf_int()
        try:
            row.append([np.round(np.exp(conf).iloc[1, 0], 2), np.round(np.exp(conf).iloc[1, 1], 2)])
        except:
            row.append([np.round(np.exp(conf).iloc[0, 0], 2), np.round(np.exp(conf).iloc[0, 1], 2)])
        # p-value
        try:
            row.append(result.pvalues[1].round(3))
        except:
            row.append(result.pvalues[0].round(3))

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
        data contains variables and la bel values

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
    y_pred_list = []
    np.random.seed(0)

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
        y_pred_list.append(y_pred)

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

        # DeLong implementation
        alpha = .95
        auc_delong, auc_cov_delong = delong_roc_variance(y_test, y_pred)
        auc_std_delong = np.sqrt(auc_cov_delong)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

        ci = stats.norm.ppf(lower_upper_q, loc=auc_delong, scale=auc_std_delong)
        ci[ci > 1] = 1

        print('DeLong: ', cols)
        print('AUC: ', auc_delong)
        print('95% CI: ', ci)

        # plot multivariate analysis AUC figure
        label = feature_name_list[i] + ', AUC=' + str(np.round(auc_, 2))
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
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(os.path.join(save_dir, 'Fig_multi_AUC_{}.png'.format(fig_title)),
                dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    # DeLong:
    p1 = delong_roc_test(y_test, y_pred_list[0], y_pred_list[2])
    print("Compare DeLong between [0, 2], P_value = ", p1)
    p2 = delong_roc_test(y_test, y_pred_list[1], y_pred_list[3])
    print("Compare DeLong between [1, 3], P_value = ", p2)

    # plot incremental_association
    if not y_min:
        y_max = max(y_axis_list) + 0.5
        y_min = min(y_axis_list) - 0.5

    # plot bar figure
    feature_name_list_bar = feature_name_list.copy()
    if rotation == 0:
        for i in range(len(feature_name_list_bar)):
            if i % 2 == 0:
                feature_name_list_bar[i] = "\n" + feature_name_list_bar[i]
            else:
                feature_name_list_bar[i] = "\n\n\n" + feature_name_list_bar[i]

    x = range(len(feature_name_list_bar))
    if horizontal_bar:
        plt.barh(x, y_axis_list, color='tab:blue', alpha=0.6)
        plt.yticks(x, feature_name_list_bar)
        plt.xlim(y_min, y_max)
        for i, v in enumerate(y_axis_list):
            plt.text(v, i, '{:.02f}'.format(v), color='tab:blue', ha='left', va='center')
    else:
        plt.bar(x, y_axis_list, color='tab:blue', alpha=0.6, width=0.5)
        plt.xticks(x, feature_name_list_bar, rotation=rotation, ha='right')
        plt.ylim(y_min, y_max)
        for i, v in enumerate(y_axis_list):
            plt.text(i, v, '{:.02f}'.format(v), color='tab:blue', ha='center', va='bottom')

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
                h = y_axis_list[0] - h_step * i - 0.3

            plt.plot(v, [h, h], color='tab:red')
            plt.text(np.mean(v), h, 'lr={:.02f}, p={:.03f}'.format(lr, p), color='tab:red', ha='center', va='bottom')

    plt.grid(axis='y')
    plt.ylabel("Akaike information criterion")
    # plt.subplots_adjust(bottom=0.5)
    plt.savefig(os.path.join(save_dir, 'Fig_incremental_{}.png'.format(fig_title)), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    print("shape_list: ", shape_list)
    print("llf_list: ", llf_list)
    print()

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


def plot_bar_line(x_bar, y_bar, name_bar, save_dir, fig_title,
                  line_index_list, lr_list, p_list, line_start_from_lowest=True, horizontal_bar=False, rotation=0,
                  line_h_list=None):
    """
    :param x_bar:
    :param y_bar:
    :param name_bar:
    :param save_dir:
    :param line_index_list:
    :param df_list: degree of freedom for each model.
    :param llf_list: log likelihood list for each model.
    :param line_start_from_lowest:
    :return:
    """
    sns.set(style='ticks', color_codes=True)

    y_max = max(y_bar) + 0.5
    y_min = min(y_bar) - 0.5

    for i in range(len(name_bar)):
        if i % 2 == 0:
            name_bar[i] = "\n" + name_bar[i]
        else:
            name_bar[i] = "\n\n\n" + name_bar[i]

    x = range(len(name_bar))
    if horizontal_bar:
        plt.barh(x, y_bar, color='b', alpha=0.6)
        plt.yticks(x, name_bar)
        plt.xlim(y_min, y_max)
        for i, v in enumerate(y_bar):
            plt.text(v, i, '{:.02f}'.format(v), color='b', ha='left', va='center')
    else:
        plt.bar(x, y_bar, color='b', alpha=0.6, width=0.5)
        plt.xticks(x, name_bar, rotation=rotation)
        plt.ylim(y_min, y_max)
        for i, v in enumerate(y_bar):
            plt.text(i, v, '{:.02f}'.format(v), color='b', ha='center', va='bottom')

    # the step of the red line
    if line_index_list:
        h_step = (y_bar[0] - y_min) / len(line_index_list)

        for i, v in enumerate(line_index_list):
            lr, p = lr_list[i], p_list[i]

            # the location of the red line
            if line_h_list:
                h = line_h_list[i]
            else:
                h = y_bar[0] - h_step * i - 0.05

            plt.plot(v, [h, h], color='r')
            plt.text(np.mean(v), h, 'LR={:.02f}, P={:.03f}'.format(lr, p), color='r', ha='center', va='bottom')

    plt.grid(axis='y')
    plt.ylabel("Akaike information criterion (AIC)")
    plt.savefig(os.path.join(save_dir, 'Fig_incremental_{}.png'.format(fig_title)), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == '__main__':
    feature_name_list = [
        'Clinic (LVESV + NYHA)',
        'Clinic + PBW',
        'Clinic + LVMD AE',
        'Clinic + PBW + LVMD AE',
    ]
    x_bar = range(len(feature_name_list))
    y_bar = [165.56, 164.68, 162.04, 159.34]
    save_dir = "/home/zhuo/Desktop/CRT_autoencoder/results/1023_pm0510_2types_1fold/32_RS1/infer_train_systolicPhase_review"
    plot_bar_line(x_bar, y_bar, name_bar=feature_name_list, save_dir=save_dir, line_index_list=[[0, 1], [0, 2], [1, 3]],
                  lr_list=[2.88, 5.52, 7.33], p_list=[0.090, 0.019, 0.007], line_h_list=[162, 161, 160],
                  fig_title="review")
