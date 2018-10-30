# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:24:26 2018

@author: nzx
"""

import numpy as np
import math
import matplotlib.pyplot as plt

font = {'family' : 'serif',  
        'weight' : 'normal',  
        'size'   : 16,  
} 


def init_config():
    #设置能够正常显示中文以及负号
    plt.rcParams['font.sans-serif']=['SimHei']#显示中文
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# =============================================================================
# 对于数据做一次平滑处理
# =============================================================================
def smoothData(data):
    data_lr = np.fliplr(data)
    data_ud = np.flipud(data)
    data_lrud = np.flipud(data_lr)
    new_data = (data + data_lr + data_ud + data_lrud)/4
    return new_data


#==============================================================================
# 这个函数的主要目的是计算 RMSE
#==============================================================================
def rmse(predictions,targets):
    assert predictions.size == targets.size
    return np.sqrt(np.mean((predictions-targets)**2))

# =============================================================================
# 大气衰减率
# =============================================================================
def AirAttenuation(func_distance):
    if(func_distance<1000):
        attenuation_rate =  0.99331-0.0001176*func_distance+1.97e-8*func_distance*func_distance;
    else:
        attenuation_rate =  math.exp(-0.0001106*func_distance)
    return attenuation_rate