# -*- coding: utf-8 -*-"
"""
Created on 10/07/2021  2:46 PM
Combine the perfusion and systolic phase AE features with clinic records to do the statistic analysis.

@author: Zhuo
"""
import os
import pandas as pd

from core.statistic_analysis import set_columns, label_crt_response, crt_criteria, data_structure, univariate_analysis, \
    pearson_corr, multivariate_analysis
from core.utils import ensure_directory


def combine_analysis(perfusion_path, sysphase_path, results_path):

    df_perfusion = pd.read_csv(perfusion_path)
    df_sysphase = pd.read_csv(sysphase_path)

    df_perfusion.columns = df_perfusion.columns.str.replace("AE_", "perfusion_AE_")
    df_sysphase.columns = df_sysphase.columns.str.replace("AE_", "systolicPhase_AE_")
    cols_AE_sys = [col for col in df_sysphase if "systolicPhase_AE_" in col or "ID" in col]
    df = pd.merge(df_perfusion, df_sysphase[cols_AE_sys], how="inner", on='ID')
    # df = pd.concat([df_perfusion, df_sysphase[cols_AE_sys]], join="inner", axis=1, keys=["ID"])
    clinic_cols, cat_cols, ignore_col_list = set_columns("GUIDE")

    # AE features list
    df_cols = df.columns.values.tolist()
    AE_cols = [col for col in df_cols if "AE_" in col]

    # set the label
    df, label_str = label_crt_response(df, "EF5", "Echo")

    # data CRT criteria Preperfusion_merged_data_pathprocessing
    df = crt_criteria(df, LBBB=False, death=False)

    # data structure
    data_structure(df[clinic_cols + [label_str]], [label_str], cat_cols, sort=True, save_dir=results_path,
                   tab_title='baseline', ignore_col_list=ignore_col_list, group1_name='Response', rename_dic=None,
                   group0_name='Non-response')
    df.to_csv(os.path.join(results_path, "data.csv"), index=False)

    # Statistic analysis
    # Univariate statistic_analysis in all AE features.
    _, sig05_cols, sig10_AE_cols, _, _ = univariate_analysis(df[AE_cols + [label_str]],label_str, sort_col='P_value',
                                                             tab_title='AE_perfusion&systolicPhase',
                                                             save_dir=results_path, ignore_col_list=ignore_col_list)

    # univariate_analysis of all significant p >= 0.1 features.
    if len(sig05_cols) > 0:
        uni_cols = sig05_cols + clinic_cols + [label_str]
    else:
        uni_cols = sig10_AE_cols + clinic_cols + [label_str]
    _, sig05_cols, _, _, _ = univariate_analysis(df[uni_cols], label_str, sort_col='P_value',
                                                 tab_title='AE_perfusion&systolicPhase_and_clinic',
                                                 save_dir=results_path, ignore_col_list=ignore_col_list)
    print('Significant variables: ', sig05_cols)

    # pearson_corr to see the correlation
    sig_perfusion_AE_cols = [col for col in sig05_cols if "perfusion_AE_" in col]
    sig_systolicPhase_AE_cols = [col for col in sig05_cols if "systolicPhase_AE_" in col]

    sig_clinic_cols = [col for col in sig05_cols if "AE_" not in col]
    sig_clinic_cols = [col for col in sig_clinic_cols if "Echo" not in col]
    sig_clinic_cols = ['SPECT_pre_ESV', 'NYHA']  # todo: check EDV, ESV only get one

    pearson_corr(df[sig_perfusion_AE_cols + sig_systolicPhase_AE_cols], xrotation=90,
                 filepath=os.path.join(results_path, 'pearsonCorr_AE.png'))

    clinic_cols = [col for col in clinic_cols if "Echo" not in col]
    col_name_list = []
    for col in sig_perfusion_AE_cols + sig_systolicPhase_AE_cols + clinic_cols:
        if "perfusion_AE_" in col:
            col = col.replace("perfusion_AE_", "Perfusion AE #")
        elif "systolicPhase_AE_" in col:
            col = col.replace("systolicPhase_AE_", "Systolic phase AE #")
        elif "SPECT_pre_" in col:
            col = col.replace("SPECT_pre_", "")
        elif "ECG_pre_" in col:
            col = col.replace("ECG_pre_", "")

        if "SRscore" in col:
            col = col.replace("SRscore", "SRS")

        col_name_list.append(col)

    pearson_corr(df[sig_perfusion_AE_cols + sig_systolicPhase_AE_cols + clinic_cols], xrotation=90,
                 filepath=os.path.join(results_path, 'pearsonCorr_AE&clinic.png'), col_name_list=col_name_list)

    # multi figure 1 - systolic phase
    feature_name_list = [
        'Clinic(QRSd + LVESV)',  # 0
        'Clinic + PBW',
        'Clinic + LVMD AE',
        'Clinic + PBW + LVMD AE',
    ]
    # sig_cols = ['ECG_pre_QRSd', 'Echo_pre_EDV']
    feature_list = [
        sig_clinic_cols,  # 0
        sig_clinic_cols + ['SPECT_pre_PBW'],  # 1
        sig_clinic_cols + ["systolicPhase_AE_31"],
        sig_clinic_cols + ['SPECT_pre_PBW', "systolicPhase_AE_31"],  # 6
    ]
    multivariate_analysis(df, feature_list, feature_name_list,
                          label_str, rotation=0,  # y_min=116.0001, y_max=119.4999,
                          fig_title='AE_systolicPhase', save_dir=results_path, ignore_col_list=ignore_col_list,
                          y_axis='aic', line_index_list=[[0, 1], [0, 2], [1, 3]],
                          line_h_list=[162, 161, 160],
                          )
    print("Significant AE: ", sig_systolicPhase_AE_cols)

    # # multi figure 2 - perfusion
    # feature_name_list = [
    #     'Clinic(QRSd + EDV)',
    #     'Clinic + SRS',
    #     'Clinic + Perfusion AE',
    #
    #     # 'Clinic + PBW + SRscore',
    #     # 'Clinic + PBW + Perfusion AE',
    # ]
    # # sig_cols = ['ECG_pre_QRSd', 'Echo_pre_EDV']
    # feature_list = [
    #     sig_clinic_cols,
    #     sig_clinic_cols + ['SPECT_pre_SRscore'],  # 2
    #     sig_clinic_cols + [sig_perfusion_AE_cols[0]],
    #
    #     # sig_clinic_cols + ['SPECT_pre_PBW'] + ['SPECT_pre_SRscore'],  # 3
    #     # sig_clinic_cols + ['SPECT_pre_PBW', sig_perfusion_AE_cols[0]],
    # ]
    # multivariate_analysis(df[sig05_cols + clinic_cols + [label_str]], feature_list, feature_name_list,
    #                       label_str, rotation=0,  # y_min=116.0001, y_max=119.7,
    #                       fig_title='AE_perfusion', save_dir=results_path, ignore_col_list=ignore_col_list,
    #                       y_axis='aic')
    #
    # # multi figure 3 - all
    # feature_name_list = [
    #     'Clinic: QRSd + EDV',
    #     'Clinic + PBW + SRS',
    #
    #     'Clinic + Perfusion AE + Phase AE',
    #     'Clinic + PBW + \nPerfusion AE + Phase AE',
    # ]
    # # sig_cols = ['ECG_pre_QRSd', 'Echo_pre_EDV']
    # feature_list = [
    #     sig_clinic_cols,
    #     sig_clinic_cols + ['SPECT_pre_PBW'] + ['SPECT_pre_SRscore'],
    #
    #     sig_clinic_cols + [sig_systolicPhase_AE_cols[0], sig_perfusion_AE_cols[0]],
    #     sig_clinic_cols + ['SPECT_pre_PBW', sig_systolicPhase_AE_cols[0], sig_perfusion_AE_cols[0]],
    # ]
    # multivariate_analysis(df[sig05_cols + clinic_cols + [label_str]], feature_list, feature_name_list,
    #                       label_str, rotation=0,  # y_min=113.5, y_max=121.5,
    #                       fig_title='AE_all', save_dir=results_path, ignore_col_list=ignore_col_list,
    #                       y_axis='aic', line_index_list=[[0, 1], [0, 2], [0, 3]],
    #                       # line_h_list=[117.3, 115.3, 114.3],
    #                       )



if __name__ == '__main__':
    # perfusion_merged_data_path = "/home/zhuo/Desktop/CRT_autoencoder/results/1006_pm01_3types_Loop_3folds/" \
    #                              "64_3folds_RS1_noPost/infer_allPatients_perfusion/Merged_data.csv"
    # sysphase_merged_data_path = "/home/zhuo/Desktop/CRT_autoencoder/results/1006_pm01_3types_Loop_3folds/" \
    #                             "64_3folds_RS1_noPost/infer_allPatients_systolicPhase/Merged_data.csv"
    # results_path = "/home/zhuo/Desktop/CRT_autoencoder/results/1006_pm01_3types_Loop_3folds/64_3folds_RS1_noPost/" \
    #                "results"
    # seed = 1
    perfusion_merged_data_path = "results/1023_pm0510_2types_1fold/32_RS1/infer_train_perfusion/Merged_data.csv"
    sysphase_merged_data_path = "results/1023_pm0510_2types_1fold/32_RS1/infer_train_systolicPhase/Merged_data.csv"
    results_path = "results/1023_pm0510_2types_1fold/32_RS1/results_1026"

    ensure_directory(results_path)
    combine_analysis(perfusion_merged_data_path, sysphase_merged_data_path, results_path)

# -----------------------------------------------------------------------------------------------------------
#     seed_record_list = []
#     folder = "1023_pm0510_2types_1fold"
#     n_features = 32
#     data_set = "train"
#
#     for seed in range(3):
#         # perfusion_merged_data_path = "results/1006_pm01_3types_Loop_3folds/64_3folds_RS{}_noPost/" \
#         #                              "infer_train_perfusion/Merged_data.csv".format(seed)
#         # sysphase_merged_data_path = "results/1006_pm01_3types_Loop_3folds/64_3folds_RS{}_noPost/" \
#         #                             "infer_train_systolicPhase/Merged_data.csv".format(seed)
#         # results_path = "/home/zhuo/Desktop/CRT_autoencoder/results/1006_pm01_3types_Loop_3folds/64_3folds_RS{}_noPost/" \
#         #                "results_train".format(seed)
#
#         perfusion_merged_data_path = "results/{}/{}_RS{}/infer_{}_perfusion/Merged_data.csv".format(folder, n_features, seed, data_set)
#         sysphase_merged_data_path = "results/{}/{}_RS{}/infer_{}_systolicPhase/Merged_data.csv".format(folder, n_features, seed, data_set)
#         results_path = "results/{}/{}_RS{}/results_{}".format(folder, n_features, seed, data_set)
#
#         ensure_directory(results_path)
#         try:
#             combine_analysis(perfusion_merged_data_path, sysphase_merged_data_path, results_path)
#             seed_record_list.append(seed)
#         except:
#             pass
#
#     print(seed_record_list)
