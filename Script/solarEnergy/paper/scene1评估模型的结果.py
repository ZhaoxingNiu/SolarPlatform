# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 19:01:33 2018

@author: nzx

@description：评估模型的结果
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




if __name__ == "__main__":

    scene_num = 1
    ray_num = 102400
    
    list_peak_rate = []
    list_total_rate = []
    list_rmse = []
    
    for helios_index in range(1):
    
        raytracing_path = globalVar.DATA_PATH + "../paper/scene{}/raytracing/{}/equinox_12_#{}.txt".format(scene_num,ray_num,helios_index)
        
        #model   unizar  hflcal 分别切换指标
        conv_model_path = globalVar.DATA_PATH + "../paper/scene{}/model/equinox_12_#{}.txt".format(scene_num,helios_index)
        
        helios_flux = np.genfromtxt(raytracing_path,delimiter=',')
        model_res = np.genfromtxt(conv_model_path,delimiter=' ')
        #model_res = np.fliplr(model_res)
        model_res = np.rot90(model_res,1,(1,0))
        
        raytracing_peak = helios_flux.max()
        raytracing_total = helios_flux.sum()/400
        
        model_peak = model_res.max()
        model_total = model_res.sum()/400
        rmse = myUtils.rmse(helios_flux,model_res)
        
        
        peak_rate = (model_peak - raytracing_peak)/raytracing_peak
        total_rate = (model_total - raytracing_total)/raytracing_total
        
        list_peak_rate.append(peak_rate)
        list_total_rate.append(total_rate)
        list_rmse.append(rmse)
        
        
    avg_model_peak_rate = abs(np.array(list_peak_rate)).mean()
    avg_model_total_rate =  abs(np.array(list_total_rate)).mean()
    avg_rmse = abs(np.array(list_rmse)).mean()