# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 16:05:26 2018

@author: nzx

@description: 主要用于描述光线跟踪结果的稳定性
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


scene_num = 1
"""
res_peak = dict()
res_peak[1024] = []
res_peak[2048] = []
res_peak[4096] = []
res_peak[8192] = []
res_peak[16384] = []
res_peak[32768] = []
res_peak[102400] = []

res_total = dict()
res_total[1024] = []
res_total[2048] = []
res_total[4096] = []
res_total[8192] = []
res_total[16384] = []
res_total[32768] = []
res_total[102400] = []


for ray_num in [1024,2048,4096,8192,16384,32768,102400]:
    for helios_index in range(40):
        raytracing_path = globalVar.DATA_PATH + "../paper/scene{}/raytracing/{}/equinox_12_#{}.txt".format(scene_num,ray_num,helios_index)
        
        helios_flux = np.genfromtxt(raytracing_path,delimiter=',')
        
        #plt.imshow(helios_flux, interpolation='bilinear',origin='lower', \
        #               cmap = cm.jet, vmin=0)
        #plt.colorbar()

        
        peak_val = helios_flux.max()
        total_val = helios_flux.sum()/400 
        
        res_peak[ray_num].append(peak_val)
        res_total[ray_num].append(total_val)
        
"""
        
# 分析场景一的耗时  主要分析峰值 以及  总能量的结果
res_peak_rate = dict()
res_peak_rate[1024] = [(res_peak[1024][i] -  res_peak[102400][i])/res_peak[102400][i] for i in range(40)]
res_peak_rate[2048] = [(res_peak[2048][i] -  res_peak[102400][i])/res_peak[102400][i] for i in range(40)]
res_peak_rate[4096] = [(res_peak[4096][i] -  res_peak[102400][i])/res_peak[102400][i] for i in range(40)]
res_peak_rate[8192] = [(res_peak[8192][i] -  res_peak[102400][i])/res_peak[102400][i] for i in range(40)]
res_peak_rate[16384] = [(res_peak[16384][i] -  res_peak[102400][i])/res_peak[102400][i] for i in range(40)]
res_peak_rate[32768] = [(res_peak[32768][i] -  res_peak[102400][i])/res_peak[102400][i] for i in range(40)]

avg_err_peak_rate = []
avg_err_peak_rate.append(abs(np.array(res_peak_rate[1024])).mean())
avg_err_peak_rate.append(abs(np.array(res_peak_rate[2048])).mean())
avg_err_peak_rate.append(abs(np.array(res_peak_rate[4096])).mean())
avg_err_peak_rate.append(abs(np.array(res_peak_rate[8192])).mean())
avg_err_peak_rate.append(abs(np.array(res_peak_rate[16384])).mean())
avg_err_peak_rate.append(abs(np.array(res_peak_rate[32768])).mean())


res_total_rate = dict()
res_total_rate[1024] = [(res_total[1024][i] -  res_total[102400][i])/res_total[102400][i] for i in range(40)]
res_total_rate[2048] = [(res_total[2048][i] -  res_total[102400][i])/res_total[102400][i] for i in range(40)]
res_total_rate[4096] = [(res_total[4096][i] -  res_total[102400][i])/res_total[102400][i] for i in range(40)]
res_total_rate[8192] = [(res_total[8192][i] -  res_total[102400][i])/res_total[102400][i] for i in range(40)]
res_total_rate[16384] = [(res_total[16384][i] -  res_total[102400][i])/res_total[102400][i] for i in range(40)]
res_total_rate[32768] = [(res_total[32768][i] -  res_total[102400][i])/res_total[102400][i] for i in range(40)]


avg_err_total_rate = []
avg_err_total_rate.append(abs(np.array(res_total_rate[1024])).mean())
avg_err_total_rate.append(abs(np.array(res_total_rate[2048])).mean())
avg_err_total_rate.append(abs(np.array(res_total_rate[4096])).mean())
avg_err_total_rate.append(abs(np.array(res_total_rate[8192])).mean())
avg_err_total_rate.append(abs(np.array(res_total_rate[16384])).mean())
avg_err_total_rate.append(abs(np.array(res_total_rate[32768])).mean())