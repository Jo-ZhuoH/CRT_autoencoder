# -*- coding: utf-8 -*-"
"""
Created on 12/14/2020  9:00 PM


@author: Zhuo
"""
import os
import matplotlib.pyplot as plt
import matplotlib.colors as col

from core.polarmaps.color_map import color_map


def savePolarmapFig(arr, save_dir, filename, color='rgb'):
    """Save a polarmap to a png file

    Parameters
    ----------
    arr: numpy array,
        polarmap array

    save_dir: string,
        save directory

    filename: string,
        save png filename.

    color: string,
        'rgb': color polarmaps as ECTb software.
        'gray': gray polarmaps
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, ax = plt.subplots()
    if color == 'rgb':
        ax.imshow(arr, cmap=col.ListedColormap(color_map))
    elif color == 'gray':
        ax.imshow(arr, cmap='gray')
    else:
        raise Exception('The input of the color should be \'rgb\' or \'gray\'')

    fig_size = 64
    fig.set_size_inches(fig_size / 100, fig_size / 100)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.0, wspace=0.0)
    plt.margins(0, 0)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


if __name__ == '__main__':
    pass
