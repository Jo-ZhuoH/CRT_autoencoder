# -*- coding: utf-8 -*-"
"""
Created on 12/14/2020  10:03 PM


@author: Zhuo
"""
import os
import argparse
import shutil
from tqdm import tqdm


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def preprocess_folders(args):
    ensure_folder(args.dest)

    for root, dirs, files in os.walk(args.root):
        print(root)
        for file in tqdm(files):
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                if 'pre' in root:
                    filename = "pre{}.png".format(file.split('.')[0])
                elif 'post' in root:
                    filename = "post{}.png".format(file.split('.')[0])
                else:
                    raise ValueError('The root does not contain pre or post.')
                shutil.copyfile(file_path, os.path.join(args.dest, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="", help="path to the data folder root")
    parser.add_argument('--dest', default="", help="path to the destination folder")

    args = parser.parse_args()
    # args.root = "/home/zhuo/Desktop/CRT_autoencoder/data/phase/post/GUIDE_II"
    # args.dest = "/home/zhuo/Desktop/CRT_autoencoder/data/phase/train_3type"

    # tw
    args.root = "/home/zhuo/Desktop/CRT_autoencoder/data/phase/post/TW"
    args.dest = "/home/zhuo/Desktop/CRT_autoencoder/data/phase/train/TW"
    preprocess_folders(args)
