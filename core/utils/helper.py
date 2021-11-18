# -*- coding: utf-8 -*-"
"""
Created on 09/25/2021  5:44 PM


@author: Zhuo
"""
import time
import os
import numpy as np
import torch
import json
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from core.statistic_analysis import label_crt_response, crt_criteria, center_filter, data_structure, set_columns
from core.utils import get_split


def datestr(detailed=False):
    now = time.localtime()
    if detailed:
        date_str = '{:02}{:02}_{:02}{:02}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    else:
        date_str = '{:02}{:02}'.format(now.tm_mon, now.tm_mday)
    return date_str


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(epoch, checkpoint_interval, model, optimizer, val_loss, is_best, save_path):
    ensure_directory(save_path)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer
    }
    if (epoch + 1) % checkpoint_interval == 0:
        filename = '{0}/checkpoint_{1}_{2:.3f}.ckpt'.format(save_path, epoch, val_loss)
        torch.save(state, filename)

    if is_best:
        torch.save(state, '{}/BEST_checkpoint.ckpt'.format(save_path))
        return val_loss, epoch


def save_args(path, dict):
    with open(os.path.join(path, 'commandline_args.txt'), 'w') as f:
        json.dump(dict, f, indent=2)


def get_basename_threshold(polarmap_type):
    if polarmap_type == "perfusion":
        basename_threshold = 22
    elif polarmap_type == "systolicPhase":
        basename_threshold = 26
    elif polarmap_type == "wallthk":
        basename_threshold = 20
    else:
        raise ValueError("Unsupported polarmap type: %", polarmap_type)
    return basename_threshold


def get_data_paths(
        data_path: str,
        clinic_path: str,
        results_path: str,
        response_definer='EF5',
        response_source='Echo',
        n_splits=5,
        random_seed=1,
        center="GUIDE",
        include_post=False,
        polarmap_type: str = 'perfusion',
        val=True,
):
    """
    Split data into training set and validation set from the train folder.
 = 0
    # args.image_type = 'phase'
    # args.center = "GUIDE"
    # args.response_definer = 'EF5'
    # args.feature_name = 'AE'
    #
    # args.exp_name = 'best'
    # args.out_features = 8
    # args.resume =
    :param random_seed:
    :param data_path:
    :param val_split:
    :return:
    """

    if polarmap_type not in ["perfusion", "systolicPhase", "wallthk"]:
        raise ValueError("Unsupported polarmap_type: %.", polarmap_type)

    pre_files = []
    post_files = []
    for root, subFolders, files in os.walk(data_path):
        for name in files:
            # if name.endswith(".png"):
            if name.endswith("{}.png".format(polarmap_type)):
                if "pre" in name:
                    pre_files.append(os.path.join(root, name))
                elif "post" in name:
                    post_files.append(os.path.join(root, name))
                else:
                    raise ValueError("This files is neither pre nor post: %.", os.path.join(root, name))
    pre_files.sort()
    post_files.sort()
    print()
    print("The data of the AE project.")
    print("{} pre images, {} post images found for AE.".format(len(pre_files), len(post_files)))

    pre_ids = [a.replace(data_path, '') for a in pre_files]
    pre_ids = [a.split("_")[0] for a in pre_ids]
    pre_ids = [a.replace('/', '') for a in pre_ids]

    # Get the response from the clinic records.
    clinic_df = pd.read_csv(clinic_path, header=0)
    clinic_df = center_filter(clinic_df, 'ID', center)
    clinic_df, label_str = label_crt_response(clinic_df, response_definer, response_source, dropna=False)

    # stats_df for the training that keeps the pre images without CRT response record..
    stats_df = clinic_df.copy()
    stats_df['Response'] = stats_df["Response"].fillna(2)

    # Criteria
    clinic_df = crt_criteria(clinic_df, LBBB=False, death=False)
    stats_df = crt_criteria(stats_df, LBBB=False, death=False)
    # Drop the images without CRT response record.
    clinic_df = clinic_df[clinic_df[label_str].notna()]
    clinic_cols, cat_cols, ignore_col_list = set_columns(center)

    data_structure(clinic_df[clinic_cols + [label_str]], [label_str], cat_cols, sort=True, save_dir=results_path,
                   tab_title='baseline', ignore_col_list=ignore_col_list, group1_name='Response', rename_dic=None,
                   group0_name='Non-response')

    # Get the response list based on the order of the pre_files.
    pre_X = []
    pre_y = []
    no_clinic_pre_files = []
    threshold = get_basename_threshold(polarmap_type)

    for i in range(len(pre_files)):
        basename = os.path.basename(pre_files[i])
        # Get the ID from the img_path and change the ID from GUIDE_II to match the clinic records.
        if len(basename) == threshold:
            id_patient = basename.replace("pre", "pre2")
        else:
            id_patient = basename
        id_patient = id_patient.replace("pre", "").split("_")[0]

        for idx, row in stats_df.iterrows():  # here we check all pre images for the training.
            if row['ID'] == id_patient:
                if row['Response'] == 2:
                    no_clinic_pre_files.append(pre_files[i])
                    break
                else:
                    pre_X.append(pre_files[i])
                    pre_y.append(row['Response'])
                    break

    pre_X = np.asarray(pre_X)
    pre_y = np.asarray(pre_y)
    X_train = None
    X_test = None

    if n_splits == 1:
        # No validation set.
        X_train = pre_X
        X_test = pre_X
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        print(skf)
        if val:
            for train_idx, test_idx in skf.split(pre_X, pre_y):
                X_train, X_test = pre_X[train_idx], pre_X[test_idx]
                y_train, y_test = pre_y[train_idx], pre_y[test_idx]

    if include_post:
        train_paths = X_train.tolist() + post_files + no_clinic_pre_files
    else:
        train_paths = X_train.tolist() + no_clinic_pre_files

    val_paths = X_test.tolist()

    print("Training set: n={}.".format(len(train_paths)))
    print("\t Pre imgs (n={}) with response data: {}, without response data: {}."
          .format(len(train_paths) - len(post_files), len(X_train), len(no_clinic_pre_files)))

    print("Validation set: n={}".format(len(val_paths)))


    return train_paths, val_paths


if __name__ == '__main__':
    data_path = "/home/zhuo/Desktop/CRT_autoencoder/data/phase/train_GUIDEtwo_3type"
    clinic_path = "/home/zhuo/Desktop/CRT_autoencoder/data/1010_3_clinicalData_4trials.csv"
    n_splits = 3
    print("n_splits = {}".format(n_splits))
    train_paths, val_paths = get_data_paths(
        polarmap_type='systolicPhase',  # "perfusion", "systolicPhase",
        data_path=data_path,
        clinic_path=clinic_path,
        response_definer='EF5',
        response_source='Echo',
        n_splits=3,
        random_seed=0,
        results_path="/home/zhuo/Desktop/CRT_autoencoder/results/test",
        include_post=False,
    )
