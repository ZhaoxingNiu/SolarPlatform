# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 20:30:07 2018

@author: nzx

@description: 查看一些作图的结果
"""


import numpy as np
#import pickle
import sys
import os

sys.path.append('../')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import myUtils
import globalVar
import CDF
import PDF
import imageplane

import math

helios_index = 1
for helios_index1 in range(1):
    #raytracing_path = globalVar.DATA_PATH + "../paper/scene1/raytracing/1024/equinox_12_#{}.txt".format(helios_index)
    raytracing_path = globalVar.DATA_PATH + "../paper/scene1/hflcal/equinox_12_#{}.txt".format(helios_index)
    
    helios_flux = np.genfromtxt(raytracing_path,delimiter=' ')
    #helios_flux = np.genfromtxt(raytracing_path,delimiter=',')
    
    plt.imshow(helios_flux, interpolation='bilinear',origin='lower', \
                   cmap = cm.jet, vmin=0)
    plt.colorbar()
    