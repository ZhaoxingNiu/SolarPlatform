# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:43:12 2018

@author: nzx
"""

import numpy as np
import time
import math
import myUtils
import globalVar

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# 这几个函数的主要目的是进行距离的变换
# =============================================================================
def AirAttenuation(func_distance):
    if(func_distance<1000):
        attenuation_rate =  0.99331-0.0001176*func_distance+1.97e-8*func_distance*func_distance
    else:
        attenuation_rate =  math.exp(-0.0001106*func_distance)
    return attenuation_rate

PREDICT_MAX_R = globalVar.RECE_MAX_R
def EnergyTransform(func,func_distance,predict_diatance,predict_r):
    air_rate1 = AirAttenuation(func_distance)
    air_rate2 = AirAttenuation(predict_diatance)
    distance_rate = func_distance/predict_diatance
    fun_r = predict_r*distance_rate
    if fun_r>= PREDICT_MAX_R:
        fun_r = PREDICT_MAX_R
    energy = air_rate2/air_rate1*func(fun_r)
    return energy

def EnergyTransformDeriv(func,func_distance,predict_diatance,predict_r):
    air_rate1 = AirAttenuation(func_distance)
    air_rate2 = AirAttenuation(predict_diatance)
    distance_rate = func_distance/predict_diatance
    fun_r = predict_r*distance_rate
    if fun_r>= PREDICT_MAX_R:
        fun_r = PREDICT_MAX_R
    #energy = air_rate2/air_rate1*func(fun_r,1)*distance_rate
    energy = air_rate2/air_rate1*func(fun_r,1)*distance_rate
    return energy

# =============================================================================
# 通过CDF进行距离变换得到需要的PDF
# =============================================================================
def getPDFFluxTransform(fit_func,func_distance,trans_diatance,width = 10.05,height = 10.05):
    time_start = time.time()
    grid_x = round(width/globalVar.GRID_LEN)
    grid_y = round(height/globalVar.GRID_LEN)
    fit_flux = np.zeros((grid_x,grid_y))
    
    compute_threshold = globalVar.DISTANCE_THRESHOLD
    distance_rate = func_distance/trans_diatance
    #由于能量衰减的原因，需要整体变换一下
    air_rate1 = AirAttenuation(func_distance)
    air_rate2 = AirAttenuation(trans_diatance)
    attenuation_rate = air_rate2/air_rate1
    
    for x in range(grid_x):
        for y in range(grid_y):
            pos_x = -width/2 + (x+0.5)*globalVar.GRID_LEN
            pos_y = -height/2 + (y+0.5)*globalVar.GRID_LEN
            pos_r = (pos_x**2 + pos_y **2 )**0.5            
            true_pos_r = pos_r*distance_rate
            #计算flux_map
            if pos_r <=compute_threshold:
                # 修改 compute_threshold
                fit_flux[x,y] = attenuation_rate*fit_func(true_pos_r)/math.pi/compute_threshold/compute_threshold   # 4
            else:
                # 这一部分 空气的衰减因素已经 考虑在函数中
                fit_flux[x,y] = EnergyTransformDeriv(fit_func,func_distance,trans_diatance,pos_r)/2/math.pi/pos_r
    # 修正小于0 的部分以及峰值
    fit_flux[fit_flux<0] = 0
    
    time_end = time.time()
    print("getPDFFluxTransform cost {:.6f} s ".format(time_end-time_start))
    return fit_flux

# =============================================================================
# 利用CDF得到对应的PDF
# =============================================================================
def getPDFFlux(function,width = 10.05,height = 10.05):
    grid_x = round(width/globalVar.GRID_LEN)
    grid_y = round(height/globalVar.GRID_LEN)
    fit_flux = np.zeros((grid_x,grid_y))
    
    compute_threshold = globalVar.DISTANCE_THRESHOLD
    for x in range(grid_x):
        for y in range(grid_y):
            pos_x = -width/2 + (x+0.5)*globalVar.GRID_LEN
            pos_y = -height/2 + (y+0.5)*globalVar.GRID_LEN
            pos_r = (pos_x**2 + pos_y **2 )**0.5
            #计算flux_map
            if pos_r <=compute_threshold:
                fit_flux[x,y] = function(pos_r)/math.pi/compute_threshold/compute_threshold
            else:
                fit_flux[x,y] = function(pos_r,1)/2/math.pi/pos_r

    # 修正小于0 的部分以及峰值
    fit_flux[fit_flux<0] = 0
    return fit_flux

def plot3DFluxMap(flux_map,width = 10.05,height = 10.05):
    fig3d = plt.figure('3D plot')
    ax = Axes3D(fig3d)
    grid_x = round(width/globalVar.GRID_LEN)
    grid_y = round(height/globalVar.GRID_LEN)
    x = range(grid_x)
    y = range(grid_y)
    X,Y = np.meshgrid(x,y)
    Z = flux_map[X,Y]
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.jet) #hot
    plt.show()
    

def envaluatePDF(onepoint_flux,fit_flux):
     # 计算拟合的误差
    fit_rmse = myUtils.rmse(fit_flux,onepoint_flux)
    total_energy = onepoint_flux.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    fit_total_energy = fit_flux.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    energy_rate = (fit_total_energy-total_energy)/total_energy*100
    print("PDF RMSE: {}  Total Energy: {} Fit Energy: {}  fit error： {:.2f}%".format(fit_rmse,total_energy,fit_total_energy,energy_rate))    
    # plot3DFluxMap(onepoint_flux)

# =============================================================================
# 通过CDF进行距离变换得到需要的PDF
# =============================================================================
def getPDFFluxTransform_Gaussian(fit_func,popt,func_distance,trans_diatance,width = 10.05,height = 10.05):
    time_start = time.time()
    grid_x = round(width/globalVar.GRID_LEN)
    grid_y = round(height/globalVar.GRID_LEN)
    fit_flux = np.zeros((grid_x,grid_y))
    
    compute_threshold = globalVar.DISTANCE_THRESHOLD
    distance_rate = func_distance/trans_diatance
    #由于能量衰减的原因，需要整体变换一下
    air_rate1 = AirAttenuation(func_distance)
    air_rate2 = AirAttenuation(trans_diatance)
    attenuation_rate = air_rate2/air_rate1
    
    for x in range(grid_x):
        for y in range(grid_y):
            pos_x = -width/2 + (x+0.5)*globalVar.GRID_LEN
            pos_y = -height/2 + (y+0.5)*globalVar.GRID_LEN
            pos_r = (pos_x**2 + pos_y **2 )**0.5            
            true_pos_r = pos_r*distance_rate
            #计算flux_map
            if pos_r <=compute_threshold:
                fit_flux[x,y] = attenuation_rate*fit_func(true_pos_r, *popt)/math.pi/compute_threshold   # 4
            else:
                # 这一部分 空气的衰减因素已经 考虑在函数中
                fit_flux[x,y] = EnergyTransformDeriv_Gaussian(fit_func, popt ,func_distance,trans_diatance,pos_r)/2/math.pi/pos_r
    # 修正小于0 的部分以及峰值
    fit_flux[fit_flux<0] = 0
    
    time_end = time.time()
    print("getPDFFluxTransform cost {:.6f} s ".format(time_end-time_start))
    return fit_flux


def EnergyTransformDeriv_Gaussian(func,popt,func_distance,predict_diatance,predict_r):
    air_rate1 = AirAttenuation(func_distance)
    air_rate2 = AirAttenuation(predict_diatance)
    distance_rate = func_distance/predict_diatance
    fun_r = predict_r*distance_rate
    if fun_r>= PREDICT_MAX_R:
        fun_r = PREDICT_MAX_R
    energy = air_rate2/air_rate1*func(fun_r,*popt)*distance_rate
    return energy