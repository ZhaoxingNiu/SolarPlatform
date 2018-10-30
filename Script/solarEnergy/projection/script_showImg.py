# -*- coding: utf-8 -*-
"""
Created on Tue May  8 21:24:03 2018

@author: nzx
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

#设置能够正常显示中文以及负号
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


DATA_PATH = "C:/Users/Administrator/Desktop/data/"

angle0 = DATA_PATH + 'project/face_3_1_angle0_500.txt'
flux0 = np.genfromtxt(angle0,delimiter=',')


angle30 = DATA_PATH + 'project/face_3_1_angle30_500.txt'
flux30 = np.genfromtxt(angle30,delimiter=',')

angle45 = DATA_PATH + 'project/face_3_1_angle45_500.txt'
flux45  = np.genfromtxt(angle45,delimiter=',')

max_val = max(flux0.max(),flux30.max(),flux45.max())


fig1 = plt.figure('angle_transform')

plt.subplot(131)
plt.imshow(flux0, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=0,vmax=max_val)
plt.title("angle_0")
plt.colorbar()

plt.subplot(132)
plt.imshow(flux30, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=0,vmax=max_val)
plt.title("angle_30")
plt.colorbar()

plt.subplot(133)
plt.imshow(flux45, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=0,vmax=max_val)
plt.title("angle_45")
plt.colorbar()
