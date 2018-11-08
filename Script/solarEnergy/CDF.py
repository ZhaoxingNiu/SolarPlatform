# -*- coding: utf-8 -*-
"""
Created on Wed May  9 21:18:09 2018

@author: nzx
"""

import math
import numpy as np
import globalVar
import myUtils

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import time

#==============================================================================
# staticFlux 统计光斑的分布
# width,height  表示的是接收面的实际尺寸

# 返回能量的积分和对应的R
#==============================================================================
def staticFlux(data,width = 10.05 ,height = 10.05):
    time_start = time.time()
    grid_x, grid_y = data.shape
    step_x = width / grid_x
    step_y = height/ grid_y
    grid_size = step_x*step_y
    assert abs(step_x - step_y) < 0.01
    
    # 统计能量的值
    step_r = globalVar.STEP_R
    # 统计边长0.71 部分的能量就可以了，sqrt(2)/2
    energy_static  = np.zeros(int(max(width,height) * 0.75 / step_r))
    energy_static_count  = np.zeros(int(max(width,height) * 0.75 / step_r))
    energy_static_per  = np.zeros(int(max(width,height) * 0.75 / step_r))
    energy_dis = [ (i+0.5)*step_r for i in range(energy_static.size)]
    
    for x in range(grid_x):
        for y in range(grid_y):
            pos_x = -width/2 + (x+0.5)*step_x
            pox_y = -height/2 + (y+0.5)*step_y
            pos_r = (pos_x**2 + pox_y **2 )**0.5
            energy_static[math.floor(pos_r/step_r)] += data[x,y]*grid_size
            energy_static_count[math.floor(pos_r/step_r)] += 1
            
    for i in range(energy_static.size):
        if(energy_static_count[i]!=0):    
            energy_static_per[i] = energy_static[i]/energy_static_count[i]/(globalVar.GRID_LEN*globalVar.GRID_LEN)
    time_end = time.time()
    print("staticFlux cost {:.6f} s ".format(time_end-time_start))
    return energy_static,energy_static_per,energy_dis


#==============================================================================
#  计算统计能量分布，统计r内分布的总能量
#==============================================================================
def accumulateFlux(energy_static):
    energy_accu = np.zeros(energy_static.size)
    energy_sum = 0
    for i in range(energy_static.size):
        energy_sum +=  energy_static[i]
        energy_accu[i] = energy_sum
    return energy_accu

# =============================================================================
# 这个函数的主要目的就是计算得到CDF函数，然后进行求解
# fit_type  1  带权重的B样条拟合   2 B样条拟合  3 B样条插值
# =============================================================================
# 使用与scipy 0.19 拟合
from scipy.interpolate import UnivariateSpline
# 插值
from scipy.interpolate import CubicSpline
def getCDF(step_r,energy_accu,energy_static_per,fit_type =1,title='fig_cdf'):
    time_start = time.time()
    is_plot = globalVar.IS_PLOT_CDF
    # 对于 距离进行抽样 得到对应的数据
    sample_step = int(1/globalVar.STEP_R/2)  #大约半米左右取一个点
    sample_index = [ x for x in range(0,energy_accu.size,sample_step) ]
    sample_r = [step_r[x] for x in sample_index]
    sample_accu = [ energy_accu[x] for x in sample_index]
    sample_r.insert(0,0)
    sample_accu.insert(0,0)  
    # # 使用样条函数函数进行拟合 权重可以选也可以不选
    if fit_type == 1:
        step_r_weight = [1/x for x in step_r]
        fit_fun = UnivariateSpline(step_r, energy_accu, w = step_r_weight)
        plot_title = 'Bspline fitting(Weighted)'
    elif fit_type == 2:
        # fit_fun.set_smoothing_factor(100)
        fit_fun = UnivariateSpline(sample_r, sample_accu)
        plot_title = 'Bspline fitting'
    elif fit_type == 3:
        plt.plot(sample_r,sample_accu,'ro',ms=5)
        fit_fun = CubicSpline(sample_r, sample_accu)
        plot_title = 'Bspline interplote'
    def mod_fit_fun(r):
        if(r<=globalVar.DISTANCE_THRESHOLD):
            result = fit_fun(globalVar.DISTANCE_THRESHOLD)/4/globalVar.DISTANCE_THRESHOLD/globalVar.DISTANCE_THRESHOLD
        else:
            result = fit_fun(r,1)/math.pi/2/r
        return result
    
    time_end = time.time()
    print("getCDF cost {:.6f} s ".format(time_end-time_start))
    
    if is_plot:
        fig1 = plt.figure(title)
        plt.plot(step_r,energy_static_per,'k',label='true flux',lw=3)
        plt.plot(step_r,fit_fun(step_r),'b',lw =3,label= plot_title)
        plt.plot(step_r,fit_fun(step_r,1),'c',lw =3,label=plot_title+' \'')       
        # flux_line = fit_fun(step_r,1)/math.pi/2/step_r
        flux_line = [mod_fit_fun(r) for r in step_r]
        plt.plot(step_r,flux_line,'g',lw =3,label='flux fit ')    
        plt.grid(True)
        plt.title(plot_title)
        plt.xlabel(r'$d_{receiver}(m)$')
        plt.ylabel(r'$Energy(W)\ |\ Energy\ densify(W/{m^2})$')
        plt.legend(loc='upper left')
        fig1.savefig(plot_title + ".png",dpi=400)
        #plt.show()
        plt.xlim(0,8)
        #plt.ylim(0,900)
        
        # 使用RMSE来评价 误差函数
        fit_rmse = myUtils.rmse(fit_fun(step_r),energy_accu)
        print('the CDF RMSE fit_RMSE: {} '.format(fit_rmse))
    return fit_fun


# =============================================================================
# 这两个函数使用的高斯函数来对于核函数进行拟合
# 返回值是高斯核的参数, 拟合的不是累计分布，是能量值
# =============================================================================

def gaussian(x, a, sigma):
    return a/(math.sqrt(math.pi*2)*sigma)*np.exp(-(x)*(x)/2/sigma/sigma)

def getCDF_Gaussian(step_r,energy_accu,energy_static_per):
    # 对于 距离进行抽样 得到对应的数据
    sample_step = int(1/globalVar.STEP_R/3)  #大约半米左右取一个点
    sample_index = [ x for x in range(0,energy_accu.size,sample_step) ]
    sample_r = [step_r[x] for x in sample_index]
    sample_accu = [ energy_accu[x] for x in sample_index]
    sample_energy_static = [ energy_static_per[x] for x in sample_index]
    popt,pcov = curve_fit(gaussian,sample_r,sample_energy_static)
    return popt
