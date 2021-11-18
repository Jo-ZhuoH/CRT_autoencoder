# -*- coding: utf-8 -*-
"""
Created on Sun Dec 06 14:51:45 2020
Help functions for data preprocessing.
    1. get the boolean, numeric, categorical data.
    2. get the table of the data structure.

@author: Zhuo
"""
import os
import pandas as pd
import numpy as np

from collections import Counter
from scipy import stats
import statsmodels.api as sm

from core import statistic_analysis as tables


def getBoolean(df, allow_na=False):
    # Get a set of boolean features from df
    if allow_na:
        df.dropna(inplace=True)
    bool_cols = [col for col in df if (len(df[col].value_counts()) > 0) &
                 all(df[col].value_counts().index.isin([0, 1]))]
    return bool_cols


def getNumeric(df, label_cols, bool_cols, cat_cols):
    # Get a set of numeric features from df
    all_cols = df.columns.values.tolist()
    num_cols = [col for col in all_cols if col not in (label_cols + bool_cols + cat_cols)]
    return num_cols


def getCounter(df):
    # Counter each columns' value
    for col in df:
        print(col + ' shape {}'.format(Counter(df[col])))


def ttest_independent(variable, df, label_cols):
    """Independent t-test

    Parameters
    ----------
    variable: String
        One variable's name
    df: Dataframe
        The original data
    label_cols: list
        response or super_responder

    Return
    ------
    ttest: the p-value of t-test
    """
    group1 = df[df[label_cols].to_numpy().reshape((len(df),)) == 1][variable].astype(float)
    group0 = df[df[label_cols].to_numpy().reshape((len(df),)) == 0][variable].astype(float)
    ttest = sm.stats.ttest_ind(group1, group0)
    return ttest[1]


def chiSquare(X, y):
    """Chi-square test of independence and homogeneity

    Parameters
    ----------
    X: String
        Feature to do the t-test
    y: Dataframe
        The original data

    Return
    ------
    chi2: float
        Chi-square score
    p_val: float
        P-value
    """
    obs = pd.crosstab(X, y.to_numpy().reshape((len(X),))).to_numpy()
    chi2, p_val, dof, exp = stats.chi2_contingency(obs)
    return chi2, p_val


def data_structure(df, label_cols, cat_cols, sort=True, save_dir='./results', tab_title='baseLine',
                   ignore_col_list=None, group1_name='Responder', group0_name='Non-responder', rename_dic=None):
    """
    Print data structure
    T-test p-value for numerical variables
    Chi-square p-value for bool and categorical variables

    Parameters
    ----------
    df: Dataframe
        Data for t-test/chi-square

    label_cols: List
        label_cols: response or super_responder

    cat_cols: List
        Categorical columns

    sort: bool, default True
        Sort the dataframe by the name of the features

    save_dir: string
        save directory.

    tab_title: string,
        The name of the saved table title.

    ignore_col_list: list,
        The list of ignored columns.

    group1_name: string,
        The name of the group that df[label_cols] == 1.

    group0_name: string,
        The name of the group that df[label_cols] == 0.

    rename_dic: dictionary,
        The directory of renaming the variables.

    Return
    ------
    data_df: Dataframe
        The dataframe of the characteristic table.
    """
    # ignore some columns
    if rename_dic is None:
        rename_dic = {}
    if ignore_col_list is None:
        ignore_col_list = []
    df = df.drop(columns=ignore_col_list)

    # get different types of columns
    bool_cols = getBoolean(df)
    num_cols = getNumeric(df, label_cols, bool_cols, cat_cols)
    df[num_cols] = df[num_cols].astype('float32')
    all_cols = num_cols + bool_cols + cat_cols

    # Split the data into 2 groups
    group1 = df[df[label_cols].to_numpy().reshape((len(df),)) == 1][all_cols]
    group0 = df[df[label_cols].to_numpy().reshape((len(df),)) == 0][all_cols]

    # Get the data structure form
    data_list = []
    if num_cols is not None:
        for col in num_cols:
            x = [col,  # name
                 '{:.01f} \u00B1 {:.01f}'.format(np.mean(df[col].astype(float)), np.std(df[col].astype(float))),  # All
                 '{:.01f} \u00B1 {:.01f}'.format(np.mean(group1[col].astype(float)), np.std(group1[col].astype(float))),  # Group 1
                 '{:.01f} \u00B1 {:.01f}'.format(np.mean(group0[col].astype(float)), np.std(group0[col].astype(float))),  # Group 2
                 ttest_independent(col, df, label_cols).round(3),  # t-test
                 ]

            data_list.append(x)

    if bool_cols is not None:
        for col in bool_cols:

            # name
            x = [col]

            # All
            try:
                num = df[col].value_counts()[1]
            except (ValueError, Exception):
                num = 0
            x.append('{} ({:.01f}%)'.format(num, num / len(df) * 100))

            # Group 1
            try:
                num = group1[col].value_counts()[1]
            except (ValueError, Exception):
                num = 0
            x.append('{} ({:.01f}%)'.format(num, num / len(group1) * 100))

            # Group 2
            try:
                num = group0[col].value_counts()[1]
            except (ValueError, Exception):
                num = 0
            x.append('{} ({:.01f}%)'.format(num, num / len(group0) * 100))

            # chi square test
            x.append(np.round(chiSquare(df[col], df[label_cols])[1], 3))
            data_list.append(x)

    if cat_cols is not None:
        for col in cat_cols:

            # variable name '' '' '' p-values
            x = [col, '', '', '', np.round(chiSquare(df[col], df[label_cols])[1], 3)]

            # First summary row
            data_list.append(x)

            # for different values in one column
            unique_list = df[col].unique()
            for i in unique_list:
                # name
                x = [col + str(i)]

                # All
                try:
                    num = df[col].value_counts()[i]
                except (ValueError, Exception):
                    num = 0
                x.append('{} ({:.01f}%)'.format(num, num / len(df) * 100))

                # Group 1
                try:
                    num = group1[col].value_counts()[i]
                except (ValueError, Exception):
                    num = 0
                x.append('{} ({:.01f}%)'.format(num, num / len(group1) * 100))

                # Group 2
                try:
                    num = group0[col].value_counts()[i]
                except (ValueError, Exception):
                    num = 0
                x.append('{} ({:.01f}%)'.format(num, num / len(group0) * 100))

                # empty p-value
                x.append('')
                data_list.append(x)

    data_df = pd.DataFrame(data_list, columns=['Variables', 'All(n={})'.format(len(df)),
                                                '{}\n(n={}, {:.01f}%)'.format(group1_name, len(group1),
                                                                             len(group1) / len(df) * 100),
                                               '{}\n(n={}, {:.01f}%)'.format(group0_name, len(group0),
                                                                             len(group0) / len(df) * 100),
                                               'P-value\n(T-test/Chi^2)'])

    # sort the values by name
    if sort:
        data_df = data_df.sort_values(by='Variables', ignore_index=True)

    # rename variables in the results table
    data_df['Variables'] = data_df['Variables'].replace(rename_dic)

    # save

    data_df.to_csv(os.path.join(save_dir, 'Baseline_{}.csv'.format(tab_title)), index=False)
    tables.render_mpl_table(data_df, os.path.join(save_dir, 'Fig_baseline_{}.png'.format(tab_title)))

    return data_df


def label_one_row(row, pre_LVEF, post_LVEF, response_definer, pre_ESV, post_ESV):
    """label CRT response by echo LVEF
    Parameters
    ----------
    row: dataframe
        one row in a dataframe for lambda methods

    pre_LVEF: string,
        the name of the pre column

    post_LVEF: string,
        the name of the post column

    pre_ESV: string,
        the name of the second pre column, should only be ESV

    post_ESV: string,
        the name of the second post column, should only be ESV

    response_definer: string,
        Definition of the CRT response, including 'EF5', 'ESV15', 'EF5_ESV15', 'EF15', 'ESV30', and 'EF15_ESV30'.
    """
    if response_definer == 'EF5':
        if (row[post_LVEF] - row[pre_LVEF]) >= 5:
            return 1
        elif (row[post_LVEF] - row[pre_LVEF]) < 5:
            return 0
        else:
            return np.nan
    elif response_definer == 'ESV15':
        if (row[post_ESV] - row[pre_ESV]) / row[pre_ESV] <= -0.15:
            return 1
        elif (row[post_ESV] - row[pre_ESV]) / row[pre_ESV] > -0.15:
            return 0
        else:
            return np.nan
    elif response_definer == 'EF5_ESV15':
        if ((row[post_LVEF] - row[pre_LVEF]) >= 5) or ((row[post_ESV] - row[pre_ESV]) / row[pre_ESV] <= -0.15):
            return 1
        elif ((row[post_LVEF] - row[pre_LVEF]) < 5) and ((row[post_ESV] - row[pre_ESV]) / row[pre_ESV] > -0.15):
            return 0
        else:
            return np.nan
    elif response_definer == 'EF15':
        if (row[post_LVEF] - row[pre_LVEF]) >= 15:
            return 1
        elif (row[post_LVEF] - row[pre_LVEF]) < 15:
            return 0
        else:
            return np.nan
    elif response_definer == 'ESV30':
        if (row[post_ESV] - row[pre_ESV]) / row[pre_ESV] <= -0.30:
            return 1
        elif (row[post_ESV] - row[pre_ESV]) / row[pre_ESV] > -0.30:
            return 0
        else:
            return np.nan
    elif response_definer == 'EF15_ESV30':
        if ((row[post_LVEF] - row[pre_LVEF]) >= 15) or ((row[post_LVEF] - row[pre_LVEF]) / row[pre_LVEF] <= -0.30):
            return 1
        elif ((row[post_LVEF] - row[pre_LVEF]) < 15) and ((row[post_LVEF] - row[pre_LVEF]) / row[pre_LVEF] > -0.30):
            return 0
        else:
            return np.nan


def label_crt_response(df, response_definer, response_source, dropna=True):
    """ Label CRT response

    Parameters
    ----------
    df: dataframe,
        data, included pre and post cols for defining CRT response.

    response_definer: string,
        The method to defining CRT response, including 'EF5', 'ESV15', 'EF5_ESV15', 'EF15', 'ESV30', and 'EF15_ESV30'.

    Returns
    -------
    df: dataframe,
        data, included CRT response column - 'Response' or 'SuperResponse'.
    label_str: string,
        return the label name.
    """
    if response_source == 'Echo':
        pre_LVEF = 'Echo_pre_LVEF'
        post_LVEF = 'Echo_post_LVEF'
        pre_ESV = 'Echo_pre_ESV'
        post_ESV = 'Echo_post_ESV'
    elif response_source == 'SPECT':
        pre_LVEF = 'SPECT_pre_LVEF'
        post_LVEF = 'SPECT_post_LVEF'
        pre_ESV = 'SPECT_pre_ESV'
        post_ESV = 'SPECT_post_ESV'
    else:
        raise ValueError("Unsupported respponse_source: %", response_source)
    df['Response'] = df.apply(lambda row: label_one_row(row, pre_LVEF, post_LVEF, response_definer,
                                                        pre_ESV, post_ESV), axis=1)
    label_str = 'Response'

    # only take the rows where Response is not nan
    if dropna:
        df = df[df[label_str].notna()]

    return df, label_str


def crt_criteria(df, LBBB=True, death=True):
    """
        0. LVEF >= 35% (Echo(core lab and centers) & SPECT(core lab and centers))
        1. Death/Hospitalization
        2. LBBB
        3. ECG_pre_QRSd >= 120
        4. Follow up date > 150 days (SPECT & Echo)
        5. Small heart (SPECT_pre_ESV < 20)
        """

    # LVEF >= 35%
    df = df.reset_index(drop=True)
    df_criteria = pd.DataFrame(columns=df.columns.values.tolist())
    for idx, row in df.iterrows():
        if (row.SPECT_pre_LVEF <= 35) or (row.Echo_pre_LVEF <= 35) \
                or (row.SPECT_pre_LVEF_Centres <= 35) \
                or (row.LVEF <= 35):  # Echo from centers form1
            idx_row = row["ID"]
            df_idx = df.index[df["ID"] == idx_row]
            if len(df_idx) > 0:
                df_criteria = pd.concat([df_criteria, df.iloc[df_idx]])

    df = df_criteria
    print("LVEF <= 35%")
    centers_ratio(df, 'ID')

    # Without pre file
    df.dropna(subset=['SPECT_pre_ESV'], inplace=True)
    print('Without pre SPECT file')
    centers_ratio(df, 'ID')

    # drop patients without ECG_pre_QRSd records
    df.dropna(subset=['ECG_pre_QRSd'], inplace=True)
    qrs_idx = df[(df['ECG_pre_QRSd'] < 120)].index
    df.drop(index=qrs_idx, inplace=True)
    print('ECG_pre_QRSd >= 120')
    centers_ratio(df, 'ID')

    # LBBB
    if LBBB:
        # only keep LBBB patients
        df = df[df.LBBB == 1]
        df = df.drop(columns=['LBBB'])
    else:
        # if there is nan value in LBBB, drop this column.
        if df['LBBB'].isnull().values.any():
            df = df.drop(columns=['LBBB'])
    print('LBBB')
    centers_ratio(df, 'ID')

    # drop NYHA == 1
    df = df[df.NYHA != 1]

    # # Follow up date > 150 days (SPECT & Echo)
    # rows_5m = df[
    #     ((pd.to_datetime(df['SPECT_post_date']) - pd.to_datetime(df['SPECT_pre_date'])).dt.days < 150) &
    #     ((pd.to_datetime(df['Echo_post_date']) - pd.to_datetime(df['Echo_pre_date'])).dt.days < 150)
    #     ].index
    #
    # # Follow up date > 150 days (SPECT & Echo)
    # # rows_5m = df[(pd.to_datetime(df['SPECT_post_date'])-pd.to_datetime(df['SPECT_pre_date'])).dt.days<150].index
    # df = df.drop(index=rows_5m)
    # # df = df.drop(columns=['SPECT_post_date', 'SPECT_pre_date', 'Echo_post_date', 'Echo_pre_date'])
    # print('Follow up date > 150 days')
    # centers_ratio(df, 'ID')


    # Death
    # Check the # of the Death
    if death:
        print('The # of Death: ', len(df[df['Death'] == 1]))
        df = df[df.Death != 1]
        # df = df.drop(columns=['Death'])
        print('Death/Hospitalization')
        centers_ratio(df, 'ID')

    # Small heart (SPECT_pre_ESV < 20)
    smallHeart_idx = df[(df['SPECT_pre_ESV'] < 20)].index
    # print(df.ix[smallHeart_idx]['SPECT_pre_ESV'])
    df.drop(index=smallHeart_idx, inplace=True)
    print('Small Heart')
    centers_ratio(df, 'ID')

    return df


def centers_ratio(df, key_str):
    """
    Get the ratio of data from different  centers
    """
    tw = sum(df[key_str].astype(str).str.len() < 4)
    nj = sum((df[key_str].astype(str).str.len() > 3) & (df[key_str].astype(str).str.len() < 8))
    iaea = sum(df[key_str].astype(str).str.len() > 8)
    print('Centers ratio patient #: ' + str(df.shape[0]))
    print('tw:nj:iaea = {0}:{1}:{2}\n'.format(tw, nj, iaea))


def center_filter(df, key_str, center_str):
    """Filter the center data"""
    if center_str == 'TW':
        df = df[df[key_str].astype(str).str.len() < 4]
    elif center_str == 'GUIDE':
        df = df[(df[key_str].astype(str).str.len() > 3) & (df[key_str].astype(str).str.len() < 8)]
    elif center_str == 'GUIDE1':
        df = df[(df[key_str].astype(str).str.len() > 3) & (df[key_str].astype(str).str.len() < 5)]
    elif center_str == 'GUIDE2':
        df = df[(df[key_str].astype(str).str.len() > 4) & (df[key_str].astype(str).str.len() < 8)]
    elif center_str == 'VISION':
        df = df[df[key_str].astype(str).str.len() > 8]
    elif center_str == 'two':
        # only keep GUIDE and VISION
        df = df[df[key_str].astype(str).str.len() > 3]
    else:
        raise ValueError('Error center = {} '.format(center_str))
    return df
