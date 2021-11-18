# -*- coding: utf-8 -*-"
"""
Created on 09/28/2021  4:51 PM


@author: Zhuo
"""
import os
import pandas as pd

from .data_structure import center_filter


def preprocessing_AEfeatures(features, feature_name, results, results_path, clinic_path, center):
    """
    Preprocessing the AE features extracted from the inference results.

    :param features:
    :param feature_name:
    :param results:
    :param results_path:
    :param clinic_path:
    :param center:
    :return:
    """
    col_name = []
    for i in range(features):
        col_name.append("{}_{}".format(feature_name, i + 1))
    df_temp = pd.DataFrame(results)
    ae_df = pd.DataFrame(df_temp[3].to_list(), columns=col_name)
    ae_df['ID'] = df_temp[1]
    ae_df.to_csv(os.path.join(results_path, 'extractedFeatures_{}.csv'.format(feature_name)),
                 index=False)

    # Combine results with clinic records
    clinic_df = pd.read_csv(clinic_path, header=0)
    # Delete post features and clear ID
    if center == 'two':
        ae_df = ae_df[~ae_df.ID.str.contains("post")]
        ae_df['ID'] = [x.replace('VISIONpre', '') for x in ae_df['ID']]
        ae_df['ID'] = [x.replace('GUIDEpre', '') for x in ae_df['ID']]
    else:
        clinic_df = center_filter(clinic_df, 'ID', center)
        ae_df = ae_df[~ae_df.ID.str.contains("post")]
        ae_df['ID'] = [x.replace('pre', '') for x in ae_df['ID']]

    print("Clinic_df.shape = {}, AE_df.shape = {}".format(clinic_df.shape, ae_df.shape))
    # Merge dataframes
    merged_df = pd.merge(clinic_df, ae_df, how="inner", on='ID')
    # Filter center data
    merged_df = center_filter(merged_df, 'ID', center)
    merged_df.to_csv(os.path.join(results_path, 'Merged_data.csv'.format(center)), index=False)

    # delete all zeros columns
    merged_df = merged_df.loc[:, (merged_df != 0).any(axis=0)]

    return merged_df


def set_columns(center):
    if center == 'VISION':
        clinic_cols = [
            'ACEI_or_ARB', 'Age', 'Gender', 'CAD', 'HTN', 'DM', 'ECG_pre_QRSd', 'NYHA',
            'MI', 'Race', 'Death',

            'SPECT_pre_LVEF', 'SPECT_pre_ESV', 'SPECT_pre_EDV',
            'SPECT_post_LVEF', 'SPECT_post_ESV',

            'SPECT_pre_PSD', 'SPECT_pre_PBW', 'SPECT_pre_SRscore',

            'SPECT_post_date', 'SPECT_pre_date',
            'Echo_post_date', 'Echo_pre_date',
        ]

        cat_cols = ['NYHA', 'Race']

        # ignore_col_list
        ignore_col_list = [
            'SPECT_post_date', 'SPECT_pre_date',
            'Echo_post_date', 'Echo_pre_date',
            'SPECT_post_LVEF', 'SPECT_post_ESV',
        ]
    elif center == 'two':
        clinic_cols = [
            'ACEI_or_ARB', 'Age', 'Gender', 'CAD', 'ECG_pre_QRSd', 'NYHA',
            'Race',

            'SPECT_pre_LVEF', 'SPECT_pre_ESV', 'SPECT_pre_EDV',
            'SPECT_post_LVEF', 'SPECT_post_ESV',

            'SPECT_pre_PSD', 'SPECT_pre_PBW', 'SPECT_pre_SRscore',

            'SPECT_post_date', 'SPECT_pre_date',
            'Echo_post_date', 'Echo_pre_date',
        ]

        cat_cols = ['NYHA', 'Race']

        # ignore_col_list
        ignore_col_list = [
            'SPECT_post_date', 'SPECT_pre_date',
            'Echo_post_date', 'Echo_pre_date',
            'SPECT_post_LVEF', 'SPECT_post_ESV', 'SPECT_pre_EDV',
        ]
    else:
        clinic_cols = [
            'Age', 'Gender', 'ECG_pre_QRSd', 'NYHA',
            # 'DM', 'CKD',

            'Echo_pre_LVEF',
            'Echo_pre_ESV', 'Echo_pre_EDV',
            'Echo_post_LVEF', 'Echo_post_ESV',

            'SPECT_pre_LVEF', 'SPECT_pre_ESV', 'SPECT_pre_EDV',
            'SPECT_pre_PSD', 'SPECT_pre_PBW',
            'SPECT_pre_SRscore',
            # 'SPECT_pre_50scar',
        ]

        cat_cols = ['NYHA']

        ignore_col_list = [
            'Echo_pre_ESV', 'Echo_pre_EDV',
            'Echo_post_LVEF', 'Echo_post_ESV',
        ]

    return clinic_cols, cat_cols, ignore_col_list
