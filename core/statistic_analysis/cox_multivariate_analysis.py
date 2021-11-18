# -*- coding: utf-8 -*-
"""
Created on Mon Dec 07 11:08:34 2020
Cox multivariate analysis by calling r packages

@author: Zhuo
"""
# %%
import os
import pandas as pd

# R
from rpy2.robjects.packages import importr
from rpy2.robjects import r
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri

pandas2ri.activate()

def cox_multivariate_analysis(data, time_str, dead_str, feature_list):
    """cox multivariate analysis


    Reference: 
    [1] https://rpy2.github.io/doc/v2.9.x/html/introduction.html

    """
    ## install the survival package
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    utils.install_packages(StrVector('survival'))
    r('install.packages(\'survival\')')

    #Load the library and example data set
    survival=importr('survival')

    time = data[time_str]
    dead = data[dead_str]

    # formula string
    for i in range(len(feature_list)):
        if i == 0:
            features_str = str(feature_list[0])
        elif i != 0:
            features_str += '+' + str(feature_list[i])
        else:
            print(feature_list)
            raise Exception("Feature list error")

    r('coxph(Surv({}, {}) ~ {}, data = data'.format(time, dead, features_str))
    results = r('summary()')
    return results


if __name__ == '__main__':
    df_read = '../data/DCM_ICM_CRT_ZH_1206.csv'
    save_dir = '/home/zhuo/Desktop/DCM_ICM_for_CRT/resutls'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = pd.read_csv(df_read, header=0)
    print('Data shape: {}'.format(df.shape))

    feature_list = ['QRSd_ms', 'hypertension_yes1']
    cox_multivariate_analysis(df, 'survivedTime_month', 'survived_yes1', feature_list)
