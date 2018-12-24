# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 20:30:07 2018

@author: nzx

@description: 作图说明聚焦镜的作用
"""


import numpy as np
#import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import myUtils
import globalVar



heliosta_nums = 4
raytracing_acc = np.zeros([200,200])

raytracing_acc2 = np.zeros([200,200])
 
for helios_index in range(4):
    raytracing_path = globalVar.DATA_PATH + "./testcpu/sub/sub_{}.txt".format(helios_index)
    raytracing_acc += np.genfromtxt(raytracing_path, delimiter=',')
    

raytracing_avg = raytracing_acc/heliosta_nums

tmp1 = np.r_[raytracing_avg[26:200],raytracing_avg[0:26]]
tmp1 = np.c_[tmp1[:,20:200],tmp1[:,0:20]]

tmp2 = np.r_[raytracing_avg[26:200],raytracing_avg[0:26]]
tmp2 = np.c_[tmp2[:,180:200],tmp2[:,0:180]]


tmp3 = np.r_[raytracing_avg[175:200],raytracing_avg[0:175]]
tmp3 = np.c_[tmp3[:,20:200],tmp3[:,0:20]]

tmp4 = np.r_[raytracing_avg[175:200],raytracing_avg[0:175]]
tmp4 = np.c_[tmp4[:,180:200],tmp4[:,0:180]]



raytracing_acc2 = tmp1 + tmp2 + tmp3  + tmp4

"""
ax1 = plt.subplot(131)
ax1.imshow(raytracing_acc, interpolation='bilinear',origin='lower', \
           cmap = cm.jet, vmin=0,vmax=1000)

ax1 = plt.subplot(132)
ax1.imshow(raytracing_acc2, interpolation='bilinear',origin='lower', \
           cmap = cm.jet, vmin=0,vmax=1000)
"""

raytracing_acc = raytracing_acc[30:170,30:170]
raytracing_acc2 = raytracing_acc2[30:170,30:170]

ax1 = plt.subplot(121)
raytracing_acc2 = np.rot90(raytracing_acc2,1,(1,0))
ax1.imshow(raytracing_acc2, interpolation='bilinear',origin='lower', \
           cmap = cm.jet, vmin=0,vmax=1000)
plt.colorbar()
ax1.axis('off')

ax2 = plt.subplot(122)
ax2.imshow(raytracing_acc, interpolation='bilinear',origin='lower', \
           cmap = cm.jet, vmin=0,vmax=1000)
ax2.axis('off')

