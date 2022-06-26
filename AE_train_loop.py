# -*- coding: utf-8 -*-"
"""
Created on 09/24/2021  4:12 PM


@author: Zhuo
"""
import argparse
import os
import pandas as pd

from core.utils.helper import get_data_paths, datestr, ensure_directory, save_args
from core.autoencoder.trainer import AEtrainer

polarmap_type_list = ["perfusion", "systolicPhase"]


def get_arguments():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--image_type', default='phase', type=str, choices=['phase', 'fft'])
    parser.add_argument('--image_path', default=None, type=str, help="Image data path")
    parser.add_argument('--results_path', type=str, help="Results path", default='results')
    parser.add_argument('--clinic_path', type=str, help="Tabular data path.",
                        default="/home/zhuo/Desktop/CRT_autoencoder/data/1010_3_clinicalData_4trials.csv")

    # GPU & CPU
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # Data
    parser.add_argument('--n_splits', type=int, default=5, help="Test split")
    parser.add_argument('--feature_name', type=str, default='AE', help="Feature name prefix")
    parser.add_argument('--polarmap_type', type=str, default=None, help="polarmap_type",
                        choices=["perfusion", "systolicPhase", "wallthk"])
    parser.add_argument('--random_seed', type=int, default='1', help="random_seed")
    parser.add_argument('--include_post', type=int, default='0', help="include_post")

    # Model
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--model', type=str, default="Linear_AE")
    parser.add_argument('--exp_name', type=str, default=None)

    # Training
    parser.add_argument('--center', type=str, default='GUIDE',
                        choices=['GUIDE', 'VISION', 'two', 'GUIDE1', 'GUIDE2'])
    parser.add_argument('--out_features', type=int, default=32)
    parser.add_argument('--loss', type=str, default="mse")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate.")
    parser.add_argument('--lr_update', type=str, default="auto")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")
    parser.add_argument('--num_epochs', type=int, default=1, help="number of iterations")
    parser.add_argument('--transform', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20, help="early stopping patience")
    parser.add_argument('--ckpt_interval', type=int, default='50', help="ckpt_interval")

    # Infer
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--response_definer', type=str, default='EF5', help="CRT response definer",
                        choices=['EF5', 'ESV15', 'EF5_ESV15'])
    parser.add_argument('--response_source', type=str, default='Echo', help="CRT response definer")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    excel_val = []
    excel_all = []
    excel_train = []

    for fold in [1]:
        for seed in range(3):  # todo: change the range of the seed
            for polarmap_type in polarmap_type_list:
                # parameters
                args = get_arguments()

                # ----------------------------------- Parameters --------------------------------------------
                args.num_epochs = 500  # todo: check
                args.n_splits = fold  # num_folds
                args.out_features = 32
                args.lr = 0.01
                args.include_post = 0
                args.polarmap_type = polarmap_type
                fold_name = "1023_pm0510_2types_{}fold".format(args.n_splits)
                args.exp_name = '{}_RS{}'.format(args.out_features, seed)
                # --------------------------------------------------------------------------------------------
                root = args.results_path + "/{}".format(fold_name)
                args.image_path = "/home/zhuo/Desktop/CRT_autoencoder/data/{}/train_GUIDEtwo_3type".format(args.image_type)

                ############
                # training #
                ############
                args.train = 1
                args.random_seed = seed

                args.results_path = os.path.join(root, args.exp_name, "training_{}".format(args.polarmap_type))
                ensure_directory(args.results_path)
                save_args(args.results_path, args.__dict__)
                print("Save results to: ", args.results_path)
                print("Training...")
                train_paths, val_paths = get_data_paths(
                    polarmap_type=args.polarmap_type,
                    data_path=args.image_path,
                    clinic_path=args.clinic_path,
                    response_definer=args.response_definer,
                    response_source=args.response_source,
                    n_splits=args.n_splits,
                    random_seed=args.random_seed,
                    results_path=args.results_path,
                    include_post=args.include_post,
                )
                trainer = AEtrainer(args)
                trainer.training(train_paths, val_paths)

                #########
                # infer #
                #########
                args.train = 0
                args.resume = os.path.join(root, args.exp_name, "training_{}".format(args.polarmap_type),
                                           "BEST_checkpoint.ckpt")
                ######################
                # infer in train set #
                ######################
                args.results_path = os.path.join(root, args.exp_name, "infer_train_{}".format(args.polarmap_type))
                ensure_directory(args.results_path)
                print("Save results to: ", args.results_path)
                print("Inference in train-set.")
                trainer = AEtrainer(args)
                trainer.infer(train_paths, multi=False)
                min_p_train = trainer.AE_min_p
                max_auc_train = trainer.AE_max_auc
                excel_train.append({"polarmap_type": polarmap_type, "seed": seed, "min_p": min_p_train,
                                    "max_auc": max_auc_train})

                ####################
                # infer in val set #
                ####################
                if args.n_splits != 1:
                    args.results_path = os.path.join(root, args.exp_name, "infer_val_{}".format(args.polarmap_type))
                    ensure_directory(args.results_path)
                    print("Save results to: ", args.results_path)

                    print("Inference in val-set.")
                    trainer = AEtrainer(args)
                    trainer.infer(val_paths, multi=False)
                    min_p_val = trainer.AE_min_p
                    max_auc_val = trainer.AE_max_auc
                    excel_val.append({"polarmap_type": polarmap_type, "seed": seed, "min_p": min_p_val,
                                      "max_auc": max_auc_val})

            df = pd.DataFrame(excel_val)
            df.to_excel(os.path.join(root, "p_values_val.xlsx"))
            # df = pd.DataFrame(excel_all)
            # df.to_excel(os.path.join(root, "p_values_all.xlsx"))
            df = pd.DataFrame(excel_train)
            df.to_excel(os.path.join(root, "p_values_train.xlsx"))

