# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:51:47 2018

@author: zhaoxingniu
"""

import numpy as np
import myUtils
import globalVar
import CDF
import PDF

import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
    prog='kernel generator', 
    usage='%(prog)s [options]',
    description="using the data to generate the kernel")
    
    parser.add_argument('true_dis',type = float, nargs='?',default = 500.0, help='the real distance')
    
    parser.add_argument('--ori_dis',type = float, default = 500.0, nargs='?', help='the origin distance')
    parser.add_argument('--angle',type = float, default = 0.0, nargs='?', help='the real angel')
    
    parser.add_argument('--step_r',type = float, default = 0.05, nargs='?', help='statistical interval')
    parser.add_argument('--grid_len',type = float, default = 0.05, nargs='?', help='grid length')
    parser.add_argument('--distance_threshold',type = float, default = 0.1, nargs='?', help='to reduce the error')
    parser.add_argument('--rece_width',type = float, default = 20.05, nargs='?', help='receiver width')
    parser.add_argument('--rece_height',type = float, default = 20.05, nargs='?', help='receiver height')
    parser.add_argument('--rece_max_r',type = float, default = 14.0, nargs='?', help='receiver max radius')
    
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    # 初始化变量
    myUtils.init_config()
    
    # 获取参数变量
    args = parse_args()
    
    globalVar.STEP_R = args.step_r
    globalVar.GRID_LEN = args.grid_len 
    globalVar.DISTANCE_THRESHOLD = args.distance_threshold 
    globalVar.RECE_WIDTH = args.rece_width 
    globalVar.RECE_HEIGHT = args.rece_height 
    globalVar.RECE_MAX_R = args.rece_max_r
    
    
    # 计算需要拟合函数
    real_distance = round(args.true_dis)
    real_angel = round(args.angle)
    
    
    #process_distance = round(real_distance/100)*100
    process_distance = round(args.ori_dis)
    process_angel = round(real_angel)
    
    # load the onepoint flux map
    print('load the Data  angle{}  and distance {}...'.format(process_angel,process_distance))
    onepoint_path = globalVar.DATA_PATH + "gen_flux_ori/{}/angle_{}.txt".format(process_distance, process_angel)
    #print(onepoint_path)
    onepoint_flux = np.genfromtxt(onepoint_path,delimiter=',')
    onepoint_flux = myUtils.smoothData(onepoint_flux)
    # becasue the incident angel,the onepoint_flux should mod
    onepoint_flux = onepoint_flux/math.cos(process_angel/2*math.pi/180)
    
    
    # 统计拟合得到CDF
    print('count the flux data...')
    energy_static,energy_static_per,step_r = CDF.staticFlux(onepoint_flux,width=globalVar.RECE_WIDTH,height = globalVar.RECE_HEIGHT)
    print('accumulate the flux energy and fit the cdf Fun')
    energy_accu = CDF.accumulateFlux(energy_static)
    
    fit_fun = CDF.getCDF(step_r,energy_accu,energy_static_per,1)
    fit_flux = PDF.getPDFFluxTransform(fit_fun,process_distance,real_distance,width=globalVar.RECE_WIDTH, height = globalVar.RECE_HEIGHT)
    crop_fit_flux = fit_flux[100:301,100:301]
    
    # 通过对应的 CDF去计算 PDF
    # 对于更近距离的光斑乘以系数，保持总能量一致，保持总能量一致
    if real_distance < process_distance:
        idea_total_val = 1000*0.88* myUtils.AirAttenuation(real_distance)
        real_tatal_val = crop_fit_flux.sum()/400
        crop_fit_flux = crop_fit_flux*idea_total_val/real_tatal_val
    
    # show the kernel
    # plt.imshow(fit_flux, interpolation='bilinear',origin='lower', \
    #               cmap =  cm.jet, vmin=0,vmax=300)
    
    save_path = globalVar.DATA_PATH + "gen_flux_dst/{}/distance_{}_angle_{}.txt".format(process_distance, real_distance, real_angel)
    print(save_path)
    np.savetxt(save_path,crop_fit_flux,fmt='%.6f')