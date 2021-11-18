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

polarmap_type_list = ["perfusion", "systolicPhase", "wallthk"]


def get_arguments():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--image_type', default='phase', type=str, choices=['phase', 'fft'])
    parser.add_argument('--image_path', type=str, help="Image data path",
                        default='/home/zhuo/Desktop/CRT_autoencoder/data/phase/train_GUIDEtwo_3type')
    parser.add_argument('--results_path', type=str, help="Results path",
                        default='results')
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

    # ----------------------------------- Parameters --------------------------------------------
    args.n_splits = 1  # num_folds
    args.random_seed = 1
    args.out_features = 64
    args.lr = 0.01
    args.num_epochs = 500
    args.polarmap_type = 'systolicPhase'  # 'perfusion'  'systolicPhase'

    # Train
    # args.train = 1
    # args.include_post = 0
    # args.transform = 0
    # args.exp_name = '{}_{}folds_RS{}_noPost'.format(args.out_features, args.n_splits, args.random_seed)
    # args.results_path = "/home/zhuo/Desktop/CRT_autoencoder/results/1022_am8_trainAll_noPost/training_{}"\
    #     .format(args.polarmap_type)

    # infer
    args.train = 0
    args.n_splits = 1  # num_folds todo:
    args.resume = "/home/zhuo/Desktop/CRT_autoencoder/results/1022_am8_trainAll_noPost/training_{}/BEST_checkpoint.ckpt".format(args.polarmap_type)
    args.results_path = "/home/zhuo/Desktop/CRT_autoencoder/results/1022_am8_trainAll_noPost/infer_{}".format(args.polarmap_type)
    args.image_path = "/home/zhuo/Desktop/CRT_autoencoder/data/{}/train_GUIDEtwo_3type".format(args.image_type)
    ensure_directory(args.results_path)
    # --------------------------------------------------------------------------------------------
    # if args.n_splits == 1:
    #     args.exp_name = 'allPatients'
    # else:
    #     args.exp_name = 'val'
    # date_str = datestr()
    # args.image_path = "/home/zhuo/Desktop/CRT_autoencoder/data/{}/train_GUIDEtwo_3type".format(args.image_type)
    #
    # if args.train == 1:
    #     if args.exp_name:
    #         args.exp_name = date_str+"_"+args.center+"_"+args.image_type+"_tsfm"+str(args.transform)+"_"+args.exp_name
    #         args.results_path = os.path.join(args.results_path, args.exp_name, "training")
    #     else:
    #         args.exp_name = date_str+"_"+args.center+"_"+args.image_type+"_tsfm"+str(args.transform)+"_"\
    #                         +str(args.out_features)
    #         args.results_path = os.path.join(args.results_path, args.exp_name, "training")
    #     ensure_directory(args.results_path)
    #     save_args(args.results_path, args.__dict__)
    #     print("Save results to: ", args.results_path)
    # elif args.train == 0:
    #     if not args.resume:
    #         raise ValueError("Please define the resume path.")
    #     else:
    #         infer_folder = args.resume.split('/')[-3]
    #         args.results_path = os.path.join(args.results_path, infer_folder,
    #                                          "infer_" + args.center + "_" + args.response_definer + "_" + args.exp_name)
    #     ensure_directory(args.results_path)
    #     print("Save results to: ", args.results_path)
    # else:
    #     raise ValueError("Unsupported train.")

    ensure_directory(args.results_path)
    return args


def main():
    args = get_arguments()

    if args.train:
        # Data paths
        print("Training...")
        train_paths, val_paths = get_data_paths(
            data_path=args.image_path,
            clinic_path=args.clinic_path,
            response_definer=args.response_definer,
            response_source=args.response_source,
            n_splits=args.n_splits,
            random_seed=args.random_seed,
            results_path=args.results_path,
            include_post=args.include_post,
            polarmap_type=args.polarmap_type,
        )
        trainer = AEtrainer(args)
        trainer.training(train_paths, val_paths)

    else:
        print("Infer on val set!")
        _, infer_paths = get_data_paths(
            data_path=args.image_path,
            clinic_path=args.clinic_path,
            response_definer=args.response_definer,
            response_source=args.response_source,
            n_splits=args.n_splits,
            random_seed=args.random_seed,
            results_path=args.results_path,
            include_post=args.include_post,
            polarmap_type=args.polarmap_type,
        )
        trainer = AEtrainer(args)
        trainer.infer(infer_paths)


if __name__ == '__main__':
    main()

