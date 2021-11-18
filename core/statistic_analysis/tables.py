# -*- coding: utf-8 -*-
"""
Created on Sun Dec 06 15:04:15 2020


@author: Zhuo
"""
import numpy as np
import matplotlib.pyplot as plt
import six
import os


def render_mpl_table(data, save_dir, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if len(data) != 0:
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        plt.savefig(save_dir, dpi=200, bbox_inches="tight")
        # plt.show()
        plt.close()

        return ax
    else:
        pass


def generate_latex_3lines_table():
    """Generate Latex talbes.

    
    Reference: 
    [1] https://towardsdatascience.com/how-to-create-latex-tables-directly-from-python-code-5228c5cea09a

    """
    pass
