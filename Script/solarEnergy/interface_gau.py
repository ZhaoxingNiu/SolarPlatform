# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:48:05 2018

@author: nzx
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
    prog='kernel generator(gaussian kernel)', 
    usage='%(prog)s [options]',
    description="using the data to generate the kernel")
    
    parser.add_argument('true_dis',type = float, default = 500, nargs='?', help='the real distance')
    
    parser.add_argument('--ori_dis',type = float, default = 500.0, nargs='?', help='the origin distance')
    parser.add_argument('--angel',type = float, default = 0.0, nargs='?', help='the real angel')
    
    parser.add_argument('--step_r',type = float, default = 0.05, nargs='?', help='statistical interval')
    parser.add_argument('--grid_len',type = float, default = 0.05, nargs='?', help='grid length')
    parser.add_argument('--distance_threshold',type = float, default = 0.1, nargs='?', help='to reduce the error')
    parser.add_argument('--rece_width',type = float, default = 10.05, nargs='?', help='receiver width')
    parser.add_argument('--rece_height',type = float, default = 10.05, nargs='?', help='receiver height')
    parser.add_argument('--rece_max_r',type = float, default = 7.0, nargs='?', help='receiver max radius')
    
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    # 初始化变量
    myUtils.init_config()
    
    # 获取参数变量
    args = parse_args()
    globalVar.re_init_para(args)
    
    # 计算需要拟合函数
    real_distance = round(args.true_dis)
    real_angel = round(args.angel)
    
    
    process_distance = round(real_distance/100)*100
    process_angel = round(real_angel)
    
    # load the onepoint flux map
    print('load the Data  angle{}  and distance {}...'.format(process_angel,process_distance))
    onepoint_path = globalVar.DATA_PATH + "/onepoint/{}/onepoint_angle_{}_distance_{}.txt".format(process_distance, process_angel, process_distance)
    print(onepoint_path)
    onepoint_flux = np.genfromtxt(onepoint_path,delimiter=',')
    onepoint_flux = myUtils.smoothData(onepoint_flux)
    # becasue the incident angel,the onepoint_flux should mod
    onepoint_flux = onepoint_flux/math.cos(process_angel/2*math.pi/180)
    
    
    # 统计拟合得到CDF
    print('count the flux data...')
    energy_static,energy_static_per,step_r = CDF.staticFlux(onepoint_flux,width=globalVar.RECE_WIDTH,height = globalVar.RECE_HEIGHT)
    print('accumulate the flux energy and fit the cdf Fun')
    energy_accu = CDF.accumulateFlux(energy_static)
    popt = CDF.getCDF_Gaussian(step_r,energy_accu,energy_static_per)
    fit_flux = PDF.getPDFFluxTransform_Gaussian(CDF.gaussian,popt,process_distance,real_distance,width=globalVar.RECE_WIDTH, height = globalVar.RECE_HEIGHT)
    
    # 通过对应的 CDF去计算 PDF
    print("calculate the RMSE")
    PDF.envaluatePDF(onepoint_flux,fit_flux)
    
    # show the kernel
    plt.imshow(fit_flux, interpolation='bilinear',origin='lower', \
                  cmap =  cm.jet, vmin=0,vmax=300)
    
    save_path = globalVar.DATA_PATH + "/gen_flux_gau/onepoint_angle_{}_distance_{}.txt".format(real_angel, real_distance)
    print(save_path)
    np.savetxt(save_path,fit_flux,fmt='%.6f')