# -*- coding: utf-8 -*-"
"""
Created on 09/25/2021  9:14 PM


@author: Zhuo
"""
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_loss_curve(train_loss_values, val_loss_values, val_interval, max_epoch, path):
    x_train = np.arange(0, max_epoch + 1)
    x_val = np.arange(0, max_epoch + 1, val_interval)

    plt.figure(tight_layout=True, dpi=200)
    plt.plot(x_train, train_loss_values, label="trainer")
    plt.plot(x_val, val_loss_values, label="val")
    plt.legend()
    plt.savefig(os.path.join(path, "loss_curve.png"), transparent=True)
    plt.close()



