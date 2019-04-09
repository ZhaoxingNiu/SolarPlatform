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


#设置能够正常显示中文以及负号
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

lim_num = 4
xmin = -1*lim_num
xmax = lim_num
ymin = -1*lim_num
ymax = lim_num

globalVar.FONTSIZE = 24
tick_font_size = 24

def get_countout_map(rt_res,conv_res,unizar_res,hflcal_res,pic_path):
    step = 0.05
    x = np.arange(-5,5,step)
    y = np.arange(-5,5,step)
    X,Y = np.meshgrid(x,y)
    
    # 使用均值滤波对结果进行处理
    #kernel = np.ones((5,5),np.float32)/25
    kernel = np.ones((3,3),np.float32)/9
    rt_res = cv.filter2D(rt_res,-1,kernel)
    conv_res = cv.filter2D(conv_res,-1,kernel)
    unizar_res = cv.filter2D(unizar_res,-1,kernel)
    hflcal_res = cv.filter2D(hflcal_res,-1,kernel)
    
    max_val = max(rt_res.max(),conv_res.max())
    peak_val = math.ceil(max_val/100)*100
    #inter_num = math.ceil(max_val/100)+1
    inter_num = 7
    seg_line = np.linspace(0,peak_val,inter_num)
    seg_line = seg_line[1:]
    
    #if len(seg_line) < 6:
    #    seg_line = np.linspace(0,peak_val,inter_num*2-1)
    
    fig = plt.figure(figsize=(20,6))
    
    plot2 = fig.add_subplot(132)
    
    # 第三个参数 可以为数组 也可以为划线的个数
    contour1 = plot2.contour(X, Y, rt_res, seg_line, linewidths = 3, colors='k', linestyles = 'dashed')
    plot2.clabel(contour1, fontsize = globalVar.FONTSIZE, colors=('k'), fmt='%.1f')
    
    contour2 = plt.contour(X, Y, unizar_res, seg_line, colors='b')
    # the second countour do not 
    #plt.clabel(contour2,fontsize = globalVar.FONTSIZE,colors=('b'),fmt='%.1f')
    
    # set the legend and the tick
    black_line = mlines.Line2D([], [], color='black', linewidth = 3,linestyle = '--', label='光线跟踪')
    blue_line = mlines.Line2D([], [], color='blue', label='UNIZAR')
    plot2.legend(handles=[black_line,blue_line],fontsize= globalVar.FONTSIZE)  
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xticks(fontsize = tick_font_size )
    plt.yticks(fontsize = tick_font_size)
    
    plot3 = fig.add_subplot(131)
    
    # 第三个参数 可以为数组 也可以为划线的个数
    contour1 = plot3.contour(X, Y, rt_res, seg_line, linewidths = 3, colors='k', linestyles = 'dashed')
    plot3.clabel(contour1, fontsize = globalVar.FONTSIZE, colors=('k'), fmt='%.1f')
    
    contour2 = plt.contour(X, Y, hflcal_res, seg_line, colors='b')
    # the second countour do not 
    #plt.clabel(contour2,fontsize = globalVar.FONTSIZE,colors=('b'),fmt='%.1f')
    
    # set the legend and the tick
    black_line = mlines.Line2D([], [], color='black', linewidth = 3,linestyle = '--', label='光线跟踪')
    blue_line = mlines.Line2D([], [], color='blue', label='HFLCAL')
    plot3.legend(handles=[black_line,blue_line],fontsize= globalVar.FONTSIZE)    
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xticks(fontsize = tick_font_size )
    plt.yticks(fontsize = tick_font_size)
    
    plot1 = fig.add_subplot(133)
    
    # 第三个参数 可以为数组 也可以为划线的个数
    contour1 = plot1.contour(X, Y, rt_res, seg_line, linewidths = 3, colors='k', linestyles = 'dashed')
    plot1.clabel(contour1, fontsize = globalVar.FONTSIZE, colors=('k'), fmt='%.1f')
    
    contour2 = plt.contour(X, Y, conv_res, seg_line, colors='b')
    # the second countour do not 
    #plt.clabel(contour2,fontsize = globalVar.FONTSIZE,colors=('b'),fmt='%.1f')
    # set the legend and the tick
    black_line = mlines.Line2D([], [], color='black', linewidth = 3,linestyle = '--', label='光线跟踪')
    blue_line = mlines.Line2D([], [], color='blue', label='数值卷积(本文)')
    plot1.legend(handles=[black_line,blue_line],fontsize= globalVar.FONTSIZE)    
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xticks(fontsize = tick_font_size )
    plt.yticks(fontsize = tick_font_size)
    
    #fig.show()
    fig.savefig(pic_path, dpi= 400)
    #plt.close()

def tmp_process():
    raytracing_path = globalVar.DATA_PATH + "../paper/scene1/raytracing/1024000/equinox_12_#9_1.txt"
    res1 = np.genfromtxt(raytracing_path,delimiter=',')  
    
    raytracing_path = globalVar.DATA_PATH + "../paper/scene1/raytracing/1024000/equinox_12_#9_2.txt"
    res2 = np.genfromtxt(raytracing_path,delimiter=',')  
    
    raytracing_path = globalVar.DATA_PATH + "../paper/scene1/raytracing/1024000/equinox_12_#9_3.txt"
    res3 = np.genfromtxt(raytracing_path,delimiter=',')  
    
    raytracing_path = globalVar.DATA_PATH + "../paper/scene1/raytracing/1024000/equinox_12_#9_4.txt"
    res4 = np.genfromtxt(raytracing_path,delimiter=',')  
    
    return  (res1+res2+res3+res4)/4
    
if __name__ == "__main__":

    scene_num = 1
    ray_num = 1024000
    
    tmp_process()
    
    for helios_index in [9]:
        pic_path =  globalVar.DATA_PATH + "../paper/scene{}/contour_res2/contour_equinox_12_#{}.pdf".format(scene_num,helios_index)
        raytracing_path = globalVar.DATA_PATH + "../paper/scene{}/raytracing/{}/equinox_12_#{}.txt".format(scene_num,ray_num,helios_index)
        #model   unizar  hflcal 分别切换指标
        conv_model_path = globalVar.DATA_PATH + "../paper/scene{}/model/equinox_12_#{}.txt".format(scene_num,helios_index)
        unizar_model_path = globalVar.DATA_PATH + "../paper/scene{}/unizar/equinox_12_#{}.txt".format(scene_num,helios_index)
        hflcal_model_path = globalVar.DATA_PATH + "../paper/scene{}/hflcal/equinox_12_#{}.txt".format(scene_num,helios_index)
        
        rt_res = np.genfromtxt(raytracing_path,delimiter=',')  
        #rt_res = process()
        conv_res = np.genfromtxt(conv_model_path,delimiter=' ')        
        unizar_res = np.genfromtxt(unizar_model_path,delimiter=' ')
        hflcal_res = np.genfromtxt(hflcal_model_path,delimiter=' ')
        
        #仅仅是旋转而已
        conv_res = np.rot90(conv_res,1,(1,0))
        unizar_res = np.rot90(unizar_res,1,(1,0))
        hflcal_res = np.rot90(hflcal_res,1,(1,0))
     
        conv_res = np.rot90(conv_res,1,(1,0))
        unizar_res = np.rot90(unizar_res,1,(1,0))
        hflcal_res = np.rot90(hflcal_res,1,(1,0))
        
        conv_res = np.rot90(conv_res,1,(1,0))
        unizar_res = np.rot90(unizar_res,1,(1,0))
        hflcal_res = np.rot90(hflcal_res,1,(1,0))
        
        
        # 左右翻转，便于作图
        rt_res = np.fliplr(rt_res)
        conv_res = np.fliplr(conv_res)
        unizar_res = np.fliplr(unizar_res)
        hflcal_res = np.fliplr(hflcal_res)

        
        get_countout_map(rt_res,conv_res,unizar_res,hflcal_res,pic_path)
    
    