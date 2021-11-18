import matlab.engine

# -*- coding: utf-8 -*-"
"""
Created on 05/14/2021  2:24 PM
Generate polarmaps from csv file with XML information.

- extract 64base data from csv file with XML information.
- decode 64base to sampling data (allSampling.mat)
- convert sampling data to polarmaps (allPolarmaps.mat)

@author: Zhuo
"""
import scipy.io

import pandas as pd

from core import polarmaps as decodePolarmaps, polarmaps as plotPolarmap

# from lib import *
if __name__ == "__main__":
    data_dir = 'data/GUIDE_II_data/guideII_XML_withID.csv'
    df = pd.read_csv(data_dir)
    pre_samplingMatDir = './data/preSampling.mat'
    # post_samplingMatDir = './data/postSampling.mat'
    pre_polarmapsMatDir = './data/prePolarmaps.mat'
    # post_polarmapsMatDir = './data/postPolarmaps.mat'

    # read ID and code columns
    # GUIDE, TW, IAEA data
    # df_pre_polarmaps = df[['ID', 'SPECT_pre_UnormalizedRawarrays',
    #                        'SPECT_pre_rphasarray', 'SPECT_pre_rthkarray'
    #                        ]].copy().dropna(subset=['SPECT_pre_UnormalizedRawarrays'])
    # df_post_polarmaps = df[['ID', 'SPECT_post_UnormalizedRawarrays',
    #                         'SPECT_post_rphasarray', 'SPECT_post_rthkarray'
    #                         ]].copy().dropna(subset=['SPECT_post_UnormalizedRawarrays'])
    # GUIDE II data
    df_pre_polarmaps = df[['ID', 'UnormalizedRawarrays',
                           'rphasarray', 'rthkarray'
                           ]].copy().dropna(subset=['UnormalizedRawarrays'])

    print('The shape of the pre XML dataframe: ', df_pre_polarmaps.shape)
    # print('The shape of the post XML dataframe: ', df_post_polarmaps.shape)

    # GUIDE, TW, IAEA data
    # Generate the sampling arrays from XMLs and save it as a .mat file ('allSampling.mat')
    # decodePolarmaps.decode64base(df_pre_polarmaps, myo_col='SPECT_pre_UnormalizedRawarrays',
    #                              sysphase_col='SPECT_pre_rphasarray', wallthk_col='SPECT_pre_rthkarray',
    #                              save_dir=pre_samplingMatDir)
    # decodePolarmaps.decode64base(df_post_polarmaps, myo_col='SPECT_post_UnormalizedRawarrays',
    #                              sysphase_col='SPECT_post_rphasarray', wallthk_col='SPECT_post_rthkarray',
    #                              save_dir=post_samplingMatDir)
    # GUIDE II data
    decodePolarmaps.decode64base(df_pre_polarmaps, myo_col='UnormalizedRawarrays',
                                 sysphase_col='rphasarray', wallthk_col='rthkarray',
                                 save_dir=pre_samplingMatDir)

    # Call MATLAB transfer the sampling data to polarmaps
    engine = matlab.engine.start_matlab()
    engine.addpath('./polarmaps', nargout=0)
    engine.generatePolarMap_Func(pre_samplingMatDir, pre_polarmapsMatDir, 'nearest', nargout=0)
    # engine.generatePolarMap_Func(post_samplingMatDir, post_polarmapsMatDir, 'nearest', nargout=0)
    # engine.exePolarmaps
    engine.exit()

    pre_mat = scipy.io.loadmat(pre_polarmapsMatDir)
    # post_mat = scipy.io.loadmat(post_polarmapsMatDir)

    plotPolarmap.transferMATtoPNG(pre_polarmapsMatDir, './data/phase/pre/GUIDE_II', color='gray')
    # plotPolarmap.transferMATtoPNG(post_polarmapsMatDir, './data/phase/post/', color='gray')
    # plotPolarmap.transferMATtoPNG(pre_polarmapsMatDir, './data/saved_polarmaps_rgb/pre/', color='rgb')
    # plotPolarmap.transferMATtoPNG(post_polarmapsMatDir, './data/saved_polarmaps_rgb/post/', color='rgb')
