# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:31:57 2018

@author: nzx

这个文件的主要功能是验证不同的卷积计算方式的效率问题
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import random
from scipy import signal
import time

plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def processAccu(data):
    data = np.fliplr(data)
    data = np.flipud(data)
    kernal_width_size,kernal_height_size = data.shape
    accu_data = np.zeros((kernal_width_size+1,kernal_height_size+1))

    time_start = time.time()
    for i in range(kernal_width_size):
        row_sum =  0
        for j in range(kernal_height_size):
            row_sum += data[i,j]
            accu_data[i+1,j+1] = row_sum +  accu_data[i,j+1]
    time_end = time.time()
    print("预处理耗时 {:.6f} s ".format(time_end-time_start))
    return accu_data
    
#写一个函数快速计算积分
def myConv(box_range,kernal_accu):
    kernal_size,_=  kernal_accu.shape
    helf_kernal_size = int((kernal_size-2)/2)
    
    width_size,height_size = box_range.shape
    conv_result =  np.zeros((width_size,height_size))
    time_start = time.time()
    for i in range(width_size):
        for j in range(height_size):
           min_x = max(-i,-helf_kernal_size) + helf_kernal_size
           min_y = max(-j,-helf_kernal_size) + helf_kernal_size
           max_x = min(width_size-1-i,helf_kernal_size) + helf_kernal_size +1 
           max_y = min(height_size-1-j,helf_kernal_size) + helf_kernal_size +1
           conv_result[i,j] =  kernal_accu[max_x,max_y] - kernal_accu[max_x,min_y] \
           - kernal_accu[min_x,max_y] +  kernal_accu[min_x,min_y]          
    time_end = time.time()
    print("计算卷积 {:.6f} s ".format(time_end-time_start))
    return conv_result
    

if __name__ == '__main__':
    kernal_size = 5
    
#    kernal = random.rand(kernal_size,kernal_size)
    kernal = np.array([[1,1,1],[1,1,1],[1,1,1]])
    #b = np.ones((kernal_size,kernal_size))
    b = np.array([[10,0,14,1,13],[12,8,15,10,4],[2,10,10,12,0],[11,7,9,6,3],[13,0,1,15,3]])
    
    
#    time_start = time.time()
#    conv_result = signal.convolve2d(b, kernal, mode='same')
#    time_end = time.time()
#    print("convolve cost {:.6f} s ".format(time_end-time_start))
    
    time_start = time.time()
    #conv_result = signal.convolve2d(b, kernal, mode='same')
    conv_fft_result1 = signal.fftconvolve(b, kernal, mode='same')
    time_end = time.time()
    print("fftconvolve cost {:.6f} s ".format(time_end-time_start))
    
#    # 第三种方式
#    kernal_accu = processAccu(kernal)
#    conv_my_result =  myConv(b,kernal_accu)