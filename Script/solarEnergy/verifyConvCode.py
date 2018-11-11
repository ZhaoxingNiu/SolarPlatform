# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:29:14 2018

@author: nzx
"""

import numpy as np
#import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    process_angle =60
    process_distance = 500
    # 首先计算使用什么角度的 sun shape 以及 具体距离  太阳光是使用的全局坐标计算的角度
    sun_ray = Vector(math.sin(process_angle*math.pi/180),0.0,math.cos(process_angle*math.pi/180))
    
    image_coor,receiver_coor,real_dis,real_angle,energy_attenuation,image_area,receiver_area = calcShape.calcShape(
            sun_ray,
            Vector(0.0,0.0,500.0),
            Vector(3.0,0.0,1.0))
    real_angle = round(real_angle)
    
    # load the onepoint flux map
    print('load the Data  angle{} and distance {}...'.format(real_angle,process_distance))
    onepoint_path = globalVar.DATA_PATH + "onepoint/{}/onepoint_angle_{}_distance_{}.txt".format(process_distance,real_angle,process_distance)
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
    
    # load the gen kernel
    test_kernel = globalVar.DATA_PATH + "gen_flux/onepoint_angle_60_distance_500.txt"
    fit_flux =  np.genfromtxt(test_kernel)
    
    np_image = imageplane.calcOnImagePlane(fit_flux,image_coor,image_area)
    np_receiver = imageplane.transformRecevier(np_image,image_coor,receiver_coor,receiver_area,energy_attenuation)
    
    print("load the ground truth")
    conv_path = globalVar.DATA_PATH + "onepoint_conv/helio_3_1_angle_{}_distance_{}.txt".format(process_angle,process_distance)
    ground_truth =  np.genfromtxt(conv_path,delimiter=',')
    #ground_truth = myUtils.smoothData(ground_truth)
    imageplane.envaluateFlux(ground_truth,np_receiver)
  
    print("******evaluate the c++ code********")
    res_path = globalVar.DATA_PATH + "testcpu/angel{}/receiver_angel_{}.txt".format(process_angle,process_angle)
    res = np.genfromtxt(res_path)
    res = np.fliplr(res)
    res = np.rot90(res,1,(1,0))
    imageplane.envaluateFlux(ground_truth,res)
    
    
    print("******evaluate the c++ code and python ********")
    imageplane.envaluateFlux(np_receiver,res)
    
    ax1 = plt.subplot(131)
    ax1.imshow(ground_truth, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax=300)
    ax2 = plt.subplot(132)
    ax2.imshow(np_receiver, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax=300)
    ax3 = plt.subplot(133)
    ax3.imshow(res, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax=300)
    
