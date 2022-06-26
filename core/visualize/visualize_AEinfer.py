# -*- coding: utf-8 -*-"
"""
Created on 09/30/2021  10:27 AM


@author: Zhuo
"""
import matplotlib.pyplot as plt
import os


def plot_comparison_AE_results(epoch, results, fold_dir, num_figs=10):
    """ Plot the comparison results of AE as the epochs increase

    Parameters
    ------
    epoch: int
        The name of the model: Linear_AE, CNN_AE
    results: list
        The results of the model
    fold_dir: string
        The directory where the results will be saved

    """
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    imgs = []
    recons = []
    names = []
    step = 1
    for k in range(0, len(results)):
        imgs.append(results[k][0])
        names.append(results[k][1])
        recons.append(results[k][2])
        if (k + 1) % num_figs == 0:
            imgs.append(results[k][0])
            names.append(results[k][1])
            recons.append(results[k][2])
            plt.figure(figsize=(num_figs, 2))
            for i, item in enumerate(imgs):
                if i >= num_figs: break
                plt.subplot(2, num_figs, i + 1, title=names[i])
                plt.imshow(item.reshape((64, 64)))
            for i, item in enumerate(recons):
                if i >= num_figs:
                    break
                plt.subplot(2, num_figs, num_figs + i + 1)
                plt.imshow(item.reshape((64, 64)))
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.savefig(os.path.join(fold_dir, 'epoch{}_{}.png'.format(epoch, step)), dpi=200)
            plt.close()
            imgs = []
            recons = []
            names = []
            step += 1

    # for k in range(0, len(results)):
    #     plt.figure(figsize=(10, 2))
    #     imgs = results[k][0]
    #     recon = results[k][2]
    #     for i, item in enumerate(imgs):
    #         if i >= 10: break
    #         plt.subplot(2, 10, i + 1)
    #         plt.imshow(item.reshape((64, 64)))
    #     for i, item in enumerate(recon):
    #         if i >= 10:
    #             break
    #         plt.subplot(2, 10, 10 + i + 1)
    #         plt.imshow(item.reshape((64, 64)))
    #     plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    #     plt.savefig(os.path.join(fold_dir, '{}.png'.format(results[k][1])), dpi=200)