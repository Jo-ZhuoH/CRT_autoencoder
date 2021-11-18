# -*- coding: utf-8 -*-"
"""
Created on 12/14/2020  7:13 PM
FFT in polarmaps.

References:
[1] https://medium.com/@hicraigchen/digital-image-processing-using-fourier-transform-in-python-bcb49424fd82

@author: Zhuo
"""
import os
import argparse
from os import listdir
from os.path import isfile, join

import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_images(path):
    """ Read images from a directory """
    files_list = [f for f in listdir(path) if isfile(join(path, f))]
    return files_list


def distance(point1, point2):
    """ Get the distance """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def idealFilterLP(D0, imgShape):
    """ Ideal low-pass filter """
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 1
    return base


def idealFilterHP(D0, imgShape):
    """ Ideal high-pass filter """
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if distance((y, x), center) < D0:
                base[y, x] = 0
    return base


def butterworthLP(D0, imgShape, n):
    """ Butterworth low-pass filter """
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base


def butterworthHP(D0, imgShape, n):
    """ Butterworth high-pass filter """
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base


def gaussianLP(D0, imgShape):
    """ Gaussian low-pass filter """
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = math.exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base


def gaussianHP(D0, imgShape):
    """ Gaussian high-pass filter """
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - math.exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base


def plot_demo(file_path, lowpass_threshold=20):
    plt.figure(figsize=(16, 5), constrained_layout=False)

    img = cv2.imread(file_path, 0)
    plt.subplot(161), plt.imshow(img, "gray"), plt.title("Original Image")

    original = np.fft.fft2(img)
    plt.subplot(162), plt.imshow(np.log(1 + np.abs(original)), "gray"), plt.title("Spectrum")

    center = np.fft.fftshift(original)
    plt.subplot(163), plt.imshow(np.log(1 + np.abs(center)), "gray"), plt.title("Centered Spectrum")

    low_pass_center = center * idealFilterLP(lowpass_threshold, img.shape)
    plt.subplot(164), plt.imshow(np.log(1 + np.abs(low_pass_center)), "gray"), plt.title(
        "Centered Spectrum \nmultiply Low Pass Filter")

    low_pass = np.fft.ifftshift(low_pass_center)
    plt.subplot(165), plt.imshow(np.log(1 + np.abs(low_pass)), "gray"), plt.title("Decentralize")

    inverse_low_pass = np.fft.ifft2(low_pass)
    plt.subplot(166), plt.imshow(np.abs(inverse_low_pass), "gray"), plt.title("Processed Image")
    plt.savefig('demo.png', dpi=300)
    plt.show()


def save_FFT_fig(arr, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save image
    fig, ax = plt.subplots()
    ax.imshow(np.log(1 + np.abs(arr)), "gray")
    fig_size = 64
    fig.set_size_inches(fig_size / 100, fig_size / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)
    plt.savefig(join(save_dir, filename))
    plt.close()


def get_FFT_dic(arr, filename):
    plist = filename.replace('.png', '').split("_")
    pdic = {
        'id': plist[0],
        '{}'.format(plist[1]): arr,
    }
    return pdic


def frequency_domain_image(read_dir, files_list, save_dir, method=None):
    """ Frequency domain image analysis

    :param read_dir: string,
        The directory of the folder containing all images.

    :param files_list: list,
        The list of all files' name.

    :param save_dir: string,
        The directory of the saved figures.

    :param method: string,
        The method to do the frequency_domain_analysis and generate spectrum images.

    :return:
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # id_list = []
    # fft_list = []
    # fft_df = pd.DataFrame(columns=['id', 'perfusion', 'systolicPhase', 'wallthk'])
    for file_name in files_list:
        # read gray image
        img = cv2.imread(join(read_dir, file_name), 0)

        # original fft
        original = np.fft.fft2(img)

        # center fft
        center = np.fft.fftshift(original)

        # save image
        save_FFT_fig(center, save_dir, file_name)

        # save data as dataframe
        # arr = np.log(1 + np.abs(center))
        # dic = get_FFT_dic(arr, file_name)
        #
        # if dic['id'] not in fft_df['id'].values.tolist():
        #     fft_df = fft_df.append(dic, ignore_index=True)
        # else:
        #     idx = fft_df.index[fft_df['id'] == dic['id']]
        #     fft_df.at[idx, list(dic)[1]] = [dic[list(dic)[1]].tolist()]

        # low-pass filter
        if method is None:
            low_pass_center = center
        elif method == 'idealLP':
            low_pass_center = center * idealFilterLP(50, img.shape)
        elif method == 'butterworthLP':
            low_pass_center = center * butterworthLP(50, img.shape, 20)
        elif method == 'gaussianLP':
            low_pass_center = center * gaussianLP(50, img.shape)
        else:
            raise ValueError('The method should only be idealLP, butterworthLP, or gaussianLP.')

        # decentralize
        low_pass = np.fft.ifftshift(low_pass_center)

        # inverse
        inverse_low_pass = np.fft.ifft2(low_pass)

    # save fft_df
    # fft_df.to_csv(join(save_dir, csv_name), index=False)
    # print('df shape: ', fft_df.shape)
    # return fft_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="", help="path to the data folder root.")
    parser.add_argument('--dest', default="", help="path to the destination folder of images.")
    args = parser.parse_args()

    # args.root = '../data/phase/pre/GUIDE'
    # args.dest = '../data/fft/pre/GUIDE'

    files_list = get_images(args.root)
    # plot_demo(join(args.root, files_list[0]), 20)
    frequency_domain_image(args.root, files_list, args.dest)
