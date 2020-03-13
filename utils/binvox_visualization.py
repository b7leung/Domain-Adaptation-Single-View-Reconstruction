# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D

# if save is false, just return the plt
def get_volume_views(volume, save_dir, n_itr, save_path=None, save = True):

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_aspect('equal')
    ax.voxels(volume, edgecolor="k")

    if save:
        if save_path is None:
            save_path = os.path.join(save_dir, 'voxels-%06d.png' % n_itr)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        return cv2.imread(save_path)
    
    else:
        return fig, ax
