# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 20:44:53 2018

@author: nzx
"""

import cv2 as cv
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm

import globalVar
import imageplane


def init_mpl():
    mpl.rcParams['xtick.labelsize'] = globalVar.FONTSIZE
    mpl.rcParams['ytick.labelsize'] = globalVar.FONTSIZE



def test_plot_map(ground_truth,res):
    ax1 = plt.subplot(121)
    ax1.imshow(ground_truth, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax=700)
    
    ax2 = plt.subplot(122)
    ax2.imshow(res, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax=700)


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
    black_line = mlines.Line2D([], [], color='black', linewidth = 3,linestyle = '--', label='Groundtruth')
    blue_line = mlines.Line2D([], [], color='blue', label='Conv model')
    plot1.legend(handles=[black_line,blue_line],fontsize= globalVar.FONTSIZE)    
    #plt.legend(handles=[contour1,contour2],labels=['Groundtruth','Conv model'],loc='best')
    fig.show()
    
    # calculate the wRMSD
    



if __name__ == '__main__':
    
    ground_truth_path = globalVar.DATA_PATH + "raytracing/shadow_test.txt"
    res_path = globalVar.DATA_PATH + "testcpu/shadow/receiver_debug.txt"
    
    init_mpl()
    print("*(*******load the ground truth******************")
    ground_truth =  np.genfromtxt(ground_truth_path,delimiter=',')
    ground_truth = np.fliplr(ground_truth)
    
    print("******evaluate the c++ code********")
    res = np.genfromtxt(res_path)
    res = np.fliplr(res)
    res = np.rot90(res,1,(1,0))
    print("********  the result  ********")
    imageplane.envaluateFlux(ground_truth,res)
    
    # plot the map 
    get_countout_map(ground_truth,res)
    
    
    
    