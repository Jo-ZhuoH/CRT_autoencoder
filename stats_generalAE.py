# -*- coding: utf-8 -*-"
"""
Created on 12/11/2020  2:39 PM


@author: Zhuo
"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_read = 'data/1211_am_Linear/echoEF5ESV15/train_Nanjing/test_Nanjing/crtData_NJ_gAE.csv'
    save_dir = 'results/generalAE/'

    df = pd.read_csv(df_read, header=0)
    print('Data shape: {}'.format(df.shape))

    # set the label
    label_str = 'response'

    # ignore_col_list
    ignore_col_list = []

    # AE feature list
    AE_cols = [col for col in df.columns.values.tolist() if "AE_" in col]
    df_AE = df[AE_cols + [label_str]]
    clinic_cols = [x for x in df.columns.values.tolist() if x not in AE_cols]

    # delete all zeros columns
    # df_AE.loc[:, (df_AE != 0).any(axis=0)]
    df_AE = df_AE.loc[:, df_AE.any()]

    # univariate_analysis of all AE features.
    _, sig05_cols, sig10_cols = univariate_analysis(df_AE, label_str, tab_title='AE features', save_dir=save_dir,
                                                    ignore_col_list=ignore_col_list, sort_col='P_value')

    print('shape of the significant AE features (P<0.05): ', len(sig05_cols))

    # pearson_corr to see the correlation with LVMD
    lvmd_cols = ['SPECT_pre_PSD', 'SPECT_pre_PBW']
    pearson_corr(df[sig05_cols + lvmd_cols], os.path.join(save_dir, 'pearsonCorr_AE_LVMD.png'))

    # univariate_analysis of AE sig 0.1 and clinic_cols
    ignore_col_list = ['HTN', 'MI', 'Concordance', 'DM', 'LBBB', 'ID', 'Race']  # these columns have nan values
    _, sig05_cols, _ = univariate_analysis(df[sig05_cols + clinic_cols], label_str,
                                           tab_title='Univariate Analysis', save_dir=save_dir,
                                           ignore_col_list=ignore_col_list, sort_col='Variables')

    # Multivariate analysis
    feature_name_list = [
        'Clinic + PSD', 'Clinic + PBW',
        'Clinic + PSD + AE #31', 'Clinic + PBW + AE #31',
        'Clinic + PSD + AE #29 #36', 'Clinic + PBW + AE #29 #36',
    ]
    sig_cols = ['Echo_pre_LVEF', 'Gender']
    feature_list = [
        sig_cols + ['SPECT_pre_PSD'],
        sig_cols + ['SPECT_pre_PBW'],
        sig_cols + ['AE_31', 'SPECT_pre_PSD'],
        sig_cols + ['AE_31', 'SPECT_pre_PBW'],
        sig_cols + ['AE_29', 'AE_36', 'SPECT_pre_PSD'],
        sig_cols + ['AE_29', 'AE_36', 'SPECT_pre_PBW'],
    ]
    multivariate_analysis(df[sig05_cols + clinic_cols], feature_list, feature_name_list, label_str, fig_title='AE',
                          save_dir=save_dir, ignore_col_list=ignore_col_list, y_axis='auc',
                          line_list=[[0, 2], [1, 3], [0, 4], [1, 5]])
