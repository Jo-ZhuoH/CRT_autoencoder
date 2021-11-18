# -*- coding: utf-8 -*-"
"""
Created on 09/25/2021  2:24 PM


@author: Zhuo
"""
from sklearn.model_selection import KFold
import numpy as np


def get_split_deterministic(all_keys, fold=0, num_splits=5, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys)):
        if i == fold:
            train_keys = np.array(all_keys)[train_idx]
            test_keys = np.array(all_keys)[test_idx]
            break
    return train_keys, test_keys


def get_split(
        all_keys,
        folds=None,
        num_splits=10,
        random_state=12345
):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param folds:
    :param num_splits:
    :param random_state:
    :return:
    """
    if folds is None:
        fold = [0, 1, 2]

    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys)):
        if i in folds:
            train_keys = np.array(all_keys)[train_idx]
            test_keys = np.array(all_keys)[test_idx]
            break
    return train_keys, test_keys
