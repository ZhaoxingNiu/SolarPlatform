# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 21:20:06 2019

@author: nzx
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 15:15:11 2019

@author: nzx
"""


import cv2 as cv
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm

import sys
import os
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import myUtils
import globalVar


#设置能够正常显示中文以及负号
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

lim_num = 5
xmin = -1*lim_num
xmax = lim_num
ymin = -1*lim_num
ymax = lim_num

globalVar.FONTSIZE = 24

tick_font_size = 18

def process_sub_file(rece_index,index_range, model_name ,delim=','):
    flux_spot = np.zeros((106,240))
    res_path = globalVar.DATA_PATH + "../paper/scene_ps10_real/{}/receiver_{}_equinox_12.txt".format(model_name,rece_index)
    for helios_index in range(index_range):
        path = globalVar.DATA_PATH + "../paper/scene_ps10_real/{}/receiver_{}_equinox_12_#{}.txt".format(model_name,rece_index,helios_index)
        helios_flux = np.genfromtxt(path,delimiter=delim)
        flux_spot += helios_flux
    np.savetxt(res_path,flux_spot,fmt='%0.4f',delimiter=delim)    
    
def getReceiverSub(rece_index,index_range, model_name ,delim=','):
    flux_spot = np.zeros((106,240))
    for helios_index in range(index_range):
        path = globalVar.DATA_PATH + "../paper/scene_ps10_real/{}/receiver_{}_equinox_12_#{}.txt".format(model_name,rece_index,helios_index)
        helios_flux = np.genfromtxt(path,delimiter=delim)
        flux_spot += helios_flux
    return flux_spot

def getReceiverTotal(rece_index, model_name ,delim=','):
    flux_spot = np.zeros((106,240))
    path = globalVar.DATA_PATH + "../paper/scene_ps10_real/{}/receiver_{}_equinox_12.txt".format(model_name,rece_index)
    helios_flux = np.genfromtxt(path,delimiter=delim)
    flux_spot += helios_flux
    # 修正总能量
    flux_spot[flux_spot<0] = 0
    return flux_spot


def plotReceFig(rece0, rece1, rece2, rece3):
    fig = plt.figure(figsize=(15,15))
    
    max_val = max(rece0.max(), rece1.max(), rece2.max(), rece3.max())
    #max_val = 950000
    
    plot1 = fig.add_subplot(141)
    plot1.imshow(rece0, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax = max_val)
    plot1.axis('off')
  
    plot2 = fig.add_subplot(142)
    plot2.imshow(rece1, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax = max_val)
    plot2.axis('off')

    
    plot3 = fig.add_subplot(143)
    plot3.imshow(rece2, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax = max_val)
    plot3.axis('off')
    
    plot4 = fig.add_subplot(144)
    plot4.imshow(rece3, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax = max_val)
    plot4.axis('off')

def init():
    helios_index_range = 250
    process_sub_file(0,helios_index_range, 'model/s1', ' ')
    process_sub_file(1,helios_index_range, 'model/s1', ' ')
    process_sub_file(2,helios_index_range, 'model/s1', ' ')
    process_sub_file(3,helios_index_range, 'model/s1', ' ')
    

if __name__ == "__main__":
    init()
    np.seterr(divide='ignore',invalid='ignore')
    receiver0 = getReceiverTotal(0, 'model/s1', ' ')
    receiver1 = getReceiverTotal(1, 'model/s1', ' ')
    receiver2 = getReceiverTotal(2, 'model/s1', ' ')
    receiver3 = getReceiverTotal(3, 'model/s1', ' ')
    
    receiver0 = np.rot90(receiver0,1,(0,1))
    receiver1 = np.rot90(receiver1,1,(0,1))
    receiver2 = np.rot90(receiver2,1,(0,1))
    receiver3 = np.rot90(receiver3,1,(0,1))
    
    sum_val = (receiver0.sum() + receiver1.sum() + receiver2.sum() + receiver3.sum())/400/1000
    peak_val = max(receiver0.max(),receiver1.max(),receiver2.max(),receiver3.max())/1000
#    receiver0 = np.flipud(receiver0)
#    receiver1 = np.flipud(receiver1)
#    receiver2 = np.flipud(receiver2)
#    receiver3 = np.flipud(receiver3)
#    receiver0 = np.rot90(receiver0,1,(1,0))
#    receiver1 = np.rot90(receiver1,1,(1,0))
#    receiver2 = np.rot90(receiver2,1,(1,0))
#    receiver3 = np.rot90(receiver3,1,(1,0))
#    
#    receiver0 = np.rot90(receiver0,1,(1,0))
#    receiver1 = np.rot90(receiver1,1,(1,0))
#    receiver2 = np.rot90(receiver2,1,(1,0))
#    receiver3 = np.rot90(receiver3,1,(1,0))
    
    plotReceFig(receiver0,receiver1,receiver2,receiver3)
