# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:04:08 2018

@author: nzx
"""


import numpy as np
#import pickle

import matplotlib.pyplot as plt
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


#设置能够正常显示中文以及负号
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        
                     
if __name__ == '__main__':
    process_angle = 60
    process_distance = 500
    
    # 首先计算使用什么角度的 sun shape 以及 具体距离  太阳光是使用的全局坐标计算的角度w
    sun_ray = Vector(0.0,-0.5, 0.866025404)
    # 计算300m定日镜
    heliostat_id = 34
    heliostat_area = 5.0008
    heliostat_pos = Vector(433,3.0,250)
    heliostat_size = Vector(2.66,0.1,1.88)
    receiver_center = Vector(0,100,0)
    receiver_x_axis = Vector(1,0,0)
    receiver_norm = Vector(0,0,1)
    
    image_coor,receiver_coor,real_dis,real_angle,energy_attenuation,image_area,receiver_area\
        = calcShape.calcShape(sun_ray,heliostat_pos,heliostat_size,receiver_center,receiver_x_axis,receiver_norm)
    real_angle = int(real_angle)
    # load the onepoint flux map
    print('load the Data  angle{}   and distance {}...'.format(real_angle,process_distance))
    onepoint_path = globalVar.DATA_PATH + "/onepoint/{}/onepoint_angle_{}_distance_{}.txt".format(process_distance,real_angle,process_distance)
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
    fit_flux = fit_flux*heliostat_area/image_area
    np_image = imageplane.calcOnImagePlane(fit_flux,image_coor,image_area)
    
    np_receiver = imageplane.transformRecevier(np_image,image_coor,receiver_coor,receiver_area,energy_attenuation)
    
# =============================================================================
#    这一段代码验证的是image plane 上的能量分布
# =============================================================================
#    print("load the ground truth")
#    conv_path = globalVar.DATA_PATH + "new/onepoint_conv/helio_3_1_angle_{}_distance_{}.txt".format(process_angle,predict_distance)
#    ground_truth =  np.genfromtxt(conv_path,delimiter=',')
#    ground_truth = myUtils.smoothData(ground_truth)
#    imageplane.envaluateFlux(ground_truth,np_receiver)
    
    print("load the ground truth")
    conv_path = globalVar.DATA_PATH + "/heliostat/heliostat_num{}_sunray_default.txt".format(heliostat_id)
    ground_truth =  np.genfromtxt(conv_path,delimiter=',')
    ground_truth = np.fliplr(ground_truth)
    imageplane.envaluateFlux(ground_truth,np_receiver)