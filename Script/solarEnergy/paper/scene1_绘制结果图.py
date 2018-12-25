# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 11:47:04 2018

@author: nzx

@description: 绘制结果图
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



def get_countout_map(ground_truth,res):
    step = 0.05
    x = np.arange(-5,5,step)
    y = np.arange(-5,5,step)
    X,Y = np.meshgrid(x,y)
    
    
    
    # 使用均值滤波对结果进行处理
    kernel = np.ones((3,3),np.float32)/9
    ground_truth = cv.filter2D(ground_truth,-1,kernel)/2
    res = cv.filter2D(res,-1,kernel)/2
    
    max_val = max(ground_truth.max(),res.max())
    peak_val = math.ceil(max_val/100)*100
    inter_num = math.ceil(max_val/100)+1
    seg_line = np.linspace(0,peak_val,inter_num)
    if len(seg_line) < 6:
        seg_line = np.linspace(0,peak_val,inter_num*2-1)
    
    
    fig = plt.figure(figsize=(10,10))
    plot1 = fig.add_subplot(111)
    
    # 第三个参数 可以为数组 也可以为划线的个数
    contour1 = plot1.contour(X, Y, ground_truth, seg_line, linewidths = 3, colors='k', linestyles = 'dashed')
    plot1.clabel(contour1, fontsize = globalVar.FONTSIZE, colors=('k'), fmt='%.1f')
    
    contour2 = plt.contour(X, Y, res, seg_line, colors='b')
    # the second countour do not 
    #plt.clabel(contour2,fontsize=10,colors=('b'),fmt='%.1f')
    
    # set the legend and the tick
    black_line = mlines.Line2D([], [], color='black', linewidth = 3,linestyle = '--', label='光线跟踪')
    blue_line = mlines.Line2D([], [], color='blue', label='Conv model')
    plot1.legend(handles=[black_line,blue_line],fontsize= globalVar.FONTSIZE)    
    #plt.legend(handles=[contour1,contour2],labels=['Groundtruth','Conv model'],loc='best')
    fig.show()
    
if __name__ == "__main__":

    scene_num = 1
    ray_num = 102400
    

    helios_index = 1
    raytracing_path = globalVar.DATA_PATH + "../paper/scene{}/raytracing/{}/equinox_12_#{}.txt".format(scene_num,ray_num,helios_index)
    #model   unizar  hflcal 分别切换指标
    conv_model_path = globalVar.DATA_PATH + "../paper/scene{}/model/equinox_12_#{}.txt".format(scene_num,helios_index)
    unizar_model_path = globalVar.DATA_PATH + "../paper/scene{}/unizar/equinox_12_#{}.txt".format(scene_num,helios_index)
    hflcal_model_path = globalVar.DATA_PATH + "../paper/scene{}/hflcal/equinox_12_#{}.txt".format(scene_num,helios_index)
    
    rt_res = 
    conv_res = 
    unizar_res = 
    hflcal_res = 
    