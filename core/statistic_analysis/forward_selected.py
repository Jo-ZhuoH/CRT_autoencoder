# -*- coding: utf-8 -*-
"""
Created on Sun Dec 06 21:49:32 2020
Forward selection with statsmodels

Reference:
[1] https://planspace.org/20150423-forward_selection_with_statsmodels/
[2] https://github.com/rasbt/mlxtend


@author: Zhuo
"""
# %%
import os
import pandas as pd
import statsmodels.formula.api as smf


# from mlxtend.feature_selection import SequentialFeatureSelector as sfs

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.logit(formula, data).fit().aic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop(0)
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.logit(formula, data).fit()
    return model


if __name__ == '__main__':
    df_read = '../data/DCM_ICM_CRT_ZH_1206.csv'
    save_dir = '/home/zhuo/Desktop/DCM_ICM_for_CRT/resutls'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = pd.read_csv(df_read, header=0)
    df = df[df['DCM'] == 0].copy()
    df = df.drop(columns=['DCM'])

    model = forward_selected(df, 'CRT_effective')
    print('Data shape: {}'.format(df.shape))
    print(model.model.formula)
    print(model.aic)
    print(model.summary())

    # prsquared dosen't stop
    # llr same
