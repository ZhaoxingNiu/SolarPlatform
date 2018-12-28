# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:39:20 2018

@author: nzx
"""


import numpy as np
#import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import myUtils
import globalVar
import CDF
import PDF
import imageplane
from mpl_toolkits.axes_grid1 import make_axes_locatable


raytracing_path_list = []
conv1_path_list = []
conv2_path_list = []

heliosta_nums = 4
raytracing_acc = np.zeros([200,200])
conv1_acc = np.zeros([200,200])
conv2_acc = np.zeros([200,200])

 
for helios_index in range(4):
    raytracing_path = globalVar.DATA_PATH + "./testcpu/sub/sub_{}.txt".format(helios_index)
    raytracing_path_list.append(raytracing_path)
    
    conv1_path = globalVar.DATA_PATH + "./testcpu/sub/conv1_sub_{}.txt".format(helios_index)
    conv1_path_list.append(conv1_path)
    
    conv2_path = globalVar.DATA_PATH + "./testcpu/sub/conv2_sub_{}.txt".format(helios_index)
    conv2_path_list.append(conv2_path)
    
    
    raytracing_acc += np.genfromtxt(raytracing_path, delimiter=',')
    
    conv1 = np.genfromtxt(conv1_path)
    conv1 = np.fliplr(conv1)
    conv1 = np.rot90(conv1,1,(1,0))
    
    conv2 = np.genfromtxt(conv2_path)
    conv2 = np.fliplr(conv2)
    conv2 = np.rot90(conv2,1,(1,0))
    
    conv1_acc += conv1
    conv2_acc += conv2
    
    
#raytracing_acc /= heliosta_nums
#conv1_acc /= heliosta_nums
#conv2_acc /= heliosta_nums
    
print("******envaluate the conv1***************\n")
imageplane.envaluateFlux(raytracing_acc,conv1_acc)

print("******envaluate the conv2***************\n")
imageplane.envaluateFlux(raytracing_acc,conv2_acc)

"""
ax1 = plt.subplot(131)
ax1.imshow(raytracing_acc, interpolation='bilinear',origin='lower', \
           cmap = cm.jet, vmin=0,vmax=300)


ax2 = plt.subplot(132)
ax2.imshow(conv1_acc, interpolation='bilinear',origin='lower', \
           cmap = cm.jet, vmin=0,vmax=300)
ax3 = plt.subplot(133)
ax3.imshow(conv2_acc, interpolation='bilinear',origin='lower', \
           cmap = cm.jet, vmin=0,vmax=300)
plt.show()
"""
ax = plt.subplot(131)
im = ax.imshow(raytracing_acc, interpolation='bilinear',origin='lower', \
           cmap =  cm.jet, vmin=0,vmax=1200)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xticks(np.linspace(0,199,11))
ax.set_xticklabels([abs(x) for x in range(-5,6,1)])
ax.set_yticks(np.linspace(0,199,11))
ax.set_yticklabels([abs(x) for x in range(-5,6,1)])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax = cax)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

ax = plt.subplot(132)
im = ax.imshow(conv1_acc, interpolation='bilinear',origin='lower', \
           cmap =  cm.jet, vmin=0,vmax=1200)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xticks(np.linspace(0,199,11))
ax.set_xticklabels([abs(x) for x in range(-5,6,1)])
ax.set_yticks(np.linspace(0,199,11))
ax.set_yticklabels([abs(x) for x in range(-5,6,1)])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax = cax)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


ax = plt.subplot(133)
im = ax.imshow(conv2_acc, interpolation='bilinear',origin='lower', \
           cmap =  cm.jet, vmin=0,vmax=1200)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_xticks(np.linspace(0,199,11))
ax.set_xticklabels([abs(x) for x in range(-5,6,1)])
ax.set_yticks(np.linspace(0,199,11))
ax.set_yticklabels([abs(x) for x in range(-5,6,1)])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax = cax)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
    
