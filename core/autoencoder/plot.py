# -*- coding: utf-8 -*-"
"""
Created on 12/15/2020  10:05 PM


@author: Zhuo
"""
import os
import matplotlib.pyplot as plt


def plot_comparison_AE_results(model_name, outputs, fold_dir):
    """ Plot the comparison results of AE as the epochs increase

    Parameters
    ------
    model_name: str
        The name of the model: Linear_AE, CNN_AE
    outputs: list
        The outputs of the training model
    fold_dir: string
        The directory where the results will be saved

    """
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    if model_name == 'Linear_AE':
        if not os.path.exists('./LinearAE_img'):
            os.mkdir('./LinearAE_img')
        for k in range(0, len(outputs)):
            plt.figure(figsize=(10, 2))
            imgs = outputs[k][1].cpu().detach().numpy()
            recon = outputs[k][2].cpu().detach().numpy()
            for i, item in enumerate(imgs):
                if i >= 10: break
                plt.subplot(2, 10, i + 1)
                plt.imshow(item.reshape((64, 64)))
            for i, item in enumerate(recon):
                if i >= 10:
                    break
                plt.subplot(2, 10, 10 + i + 1)
                plt.imshow(item.reshape((64, 64)))
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.savefig(os.path.join(fold_dir, 'epoch_{}.png'.format(outputs[k][0])), dpi=200)

    if model_name == 'CNN_AE':
        if not os.path.exists('./ConvAE_img'):
            os.mkdir('./ConvAE_img')
        for k in range(0, len(outputs)):
            plt.figure(figsize=(10, 2))
            imgs = outputs[k][1].cpu().detach().numpy()
            recon = outputs[k][2].cpu().detach().numpy()
            for i, item in enumerate(imgs):
                if i >= 10:
                    break
                plt.subplot(2, 10, i + 1)
                plt.imshow(item[0])
            for i, item in enumerate(recon):
                if i >= 10:
                    break
                plt.subplot(2, 10, 10 + i + 1)
                plt.imshow(item[0])
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.savefig(os.path.join(fold_dir, 'epoch_{}.png'.format(outputs[k][0])), dpi=200)


if __name__ == '__main__':
    pass
