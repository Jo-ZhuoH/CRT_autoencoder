# -*- coding: utf-8 -*-"
"""
Created on 12/16/2020  2:26 PM


@author: Zhuo
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_read = './data/1224pm_NJ_clinic_fft_aeData_oldVer.csv'
df = pd.read_csv(df_read, header=0)  # shape 103

df_new = 'data/1217_NJ137_clinicData.csv'  # update the response
df_new = pd.read_csv(df_new, header=0)  # shape 137

# df['Response'] = np.nan
# df['SuperResponse'] = np.nan
df['CKD'] = np.nan
df['SPECT_pre_SRscore'] = np.nan


for i, row in df.iterrows():
    key = row['ID']
    idx_df = df.index[df['ID'] == key]
    idx_new = df_new.index[df_new['ID'] == key]
    df.at[idx_df, 'CKD'] = df_new['CKD'][idx_new].values
    df.at[idx_df, 'SPECT_pre_SRscore'] = df_new['SPECT_pre_SRscore'][idx_new].values


# df = pd.merge(df, df_new, on='ID', how='left')
df.to_csv("data/1224pm_NJ_clinic_fft_aeData_oldVer.csv", index=False)

