# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:57:49 2018

@author: nzx

这个文件的主要功能是对于
"""

import numpy as np
import time
from scipy import interpolate
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

#设置能够正常显示中文以及负号
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 指定数据的位置以及输出的位置
DATA_PATH = "C:/Users/Administrator/Desktop/data/"
# 统计能量分布时 R 的bin的大小
STEP_R = 0.05
# GRID_LEN  对应的是接收面的格子边长
GRID_LEN = 0.05


# 接收面的长  接收面的宽
RECE_WIDTH = 10.05
RECE_HEIGHT = 10.05

# =============================================================================
# 选取中间的一部分像素进行插值
# 为了计算时间，我只选用中间的1m处的接收面
# =============================================================================
def get_onepoint_flux_interploate(data,width = 10.05,height = 10.05):
    # 为了运性速度，只选取中间半径为1m的矩形来进行统计
    grid_x, grid_y = data.shape
    step_x = width / grid_x
    step_y = height/ grid_y
    onepoint_1_meter = data[80:121,80:121]
    all_interploate = True
    time_start = time.time()
    newfunc = 0
    if all_interploate:
        x_pos,y_pos = np.mgrid[(-width/2 + 0.5*step_x):(width/2 + 0.5*step_x):step_x,\
                               (-height/2 + 0.5*step_y):(height/2 +0.5*step_y):step_y]
        # 插值的各种方式    nearest   zero  linear   quadratic cubic
        newfunc = interpolate.interp2d(x_pos,y_pos,data,kind = 'linear')
    else:
        width = 2.05
        height = 2.05
        x_pos,y_pos = np.mgrid[(-width/2 + 0.5*step_x):(width/2 + 0.5*step_x):step_x,\
                               (-height/2 + 0.5*step_y):(height/2 +0.5*step_y):step_y]
        # 插值的各种方式    nearest   zero  linear   quadratic cubic
        newfunc = interpolate.interp2d(x_pos,y_pos,onepoint_1_meter,kind = 'linear')
    time_end = time.time()
    print("interploate cost {} s, width = {} m".format(time_end-time_start,width/2))
    return newfunc

def get_sample_flux_interploate(func,sample_r):
    if type(sample_r) == list:
        array_len = len(sample_r)
    else:
        array_len = sample_r.size
    
    result = np.zeros(array_len)
    error = np.zeros(array_len)
    for i in range(array_len):
        r = sample_r[i]
        time_start = time.time()
        result[i],error[i] = integrate.dblquad(func, -r, r, lambda x: -(r*r-x**2)**0.5, lambda x: (r*r-x**2)**0.5)
        time_end = time.time()
        print("step:{}  interploate cost {} s ".format(i,time_end-time_start))
    return result,error


if __name__ == '__main__':
    print('load the Data...')
    onepoint_path = DATA_PATH + "onepoint_odd_distance500.txt"
    onepoint_flux = np.genfromtxt(onepoint_path,delimiter=',')
    
    newfunc = get_onepoint_flux_interploate(onepoint_flux,width = RECE_WIDTH,height = RECE_HEIGHT)
    
#    # 由于这个功能特别耗时，因此提前执行，保存到 pkl文件中
     # 这一部分用到了数值积分来进行处理
#    sample_r = [0.01*r for r in range(1,51) ]
#    sample_accu_energy,err = get_sample_flux_interploate(newfunc,sample_r)
#    np.insert(sample_accu_energy,0,0)
    
    
#    with open('onepoint_interploate_pickle.pkl', 'wb') as f:
#        pickle.dump(sample_accu_energy, f)
#        
#    with open('onepoint_interploate_pickle.pkl', 'rb') as f:
#        model_inter = pickle.load(f)
    
    with open('onepoint_interploate_fun.pkl', 'wb') as f:
        pickle.dump(newfunc, f)
        
    with open('onepoint_interploate_fun.pkl', 'rb') as f:
        onepoint_function = pickle.load(f)