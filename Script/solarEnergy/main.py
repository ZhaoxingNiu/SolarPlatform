# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:42:13 2018

@author: nzx
"""

import numpy as np
import myUtils
import globalVar
import CDF
import PDF
import imageplane

import math
import sys
sys.path.append('./projection')
import calcShape
from vector import Vector


if __name__ == '__main__':
    # 初始化变量
    myUtils.init_config()
    
    process_angle = 0
    process_distance = 500
    
    # 首先计算使用什么角度的 sun shape 以及 具体距离  太阳光是使用的全局坐标计算的角度
    sun_ray = Vector(math.sin(process_angle*math.pi/180),0.0,math.cos(process_angle*math.pi/180))
    image_coor,receiver_coor,real_dis,real_angle,energy_attenuation,image_area,receiver_area= calcShape.calcShape(sun_ray)
    real_angle = int(real_angle)
    
    # load the onepoint flux map
    print('load the Data  angle{}   and distance {}...'.format(real_angle,process_distance))
    onepoint_path = globalVar.DATA_PATH + "/onepoint/{}/onepoint_angle_{}_distance_{}.txt".format(process_distance,real_angle,process_distance)
    print(onepoint_path)
    onepoint_flux = np.genfromtxt(onepoint_path,delimiter=',')
    onepoint_flux = myUtils.smoothData(onepoint_flux)
    
    # 统计拟合得到CDF
    print('count the flux data...')
    energy_static,energy_static_per,step_r = CDF.staticFlux(onepoint_flux,width=globalVar.RECE_WIDTH,height = globalVar.RECE_HEIGHT)
    print('accumulate the flux energy and fit the cdf Fun')
    energy_accu = CDF.accumulateFlux(energy_static)
    fit_fun = CDF.getCDF(step_r,energy_accu,energy_static_per,1)
        
    # 通过对应的 CDF去计算 PDF
    print("calculate the RMSE")
    #fit_flux = PDF.getPDFFlux(fit_fun, width=globalVar.RECE_WIDTH, height = globalVar.RECE_HEIGHT)
    #PDF.envaluatePDF(onepoint_flux,fit_flux)
    fit_flux = PDF.getPDFFluxTransform(fit_fun,process_distance,real_dis,width=globalVar.RECE_WIDTH, height = globalVar.RECE_HEIGHT)
    
    # mod one_point flux energy
    heliostat_size = globalVar.HELIOSTAT_AREA
    fit_flux = fit_flux*heliostat_size/image_area
    np_image = imageplane.calcOnImagePlane(fit_flux,image_coor,image_area)
    
    np_receiver = imageplane.transformRecevier(np_image,image_coor,receiver_coor,receiver_area,energy_attenuation)
    
    print("load the ground truth")
    conv_path = globalVar.DATA_PATH + "/onepoint_conv/helio_3_1_angle_{}_distance_{}.txt".format(process_angle,process_distance)
    ground_truth =  np.genfromtxt(conv_path,delimiter=',')
    #ground_truth = myUtils.smoothData(ground_truth)
    imageplane.envaluateFlux(ground_truth,np_receiver)
    
    
    
    
    
    