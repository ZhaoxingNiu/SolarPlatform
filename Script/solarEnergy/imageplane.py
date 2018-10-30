# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:01:53 2018

@author: nzx
"""

import numpy as np
import time
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

import cv2
import globalVar
import myUtils

def calcOnImagePlane(fit_flux,image_coor,image_area):
    # 计算卷积积分
    np_image_plane = np.zeros((200,200),np.int)
    cv2.fillPoly(np_image_plane, [image_coor], 1)
    
    time_start = time.time()
    conv_fft_result = signal.fftconvolve(np_image_plane, fit_flux*globalVar.GRID_LEN*globalVar.GRID_LEN, mode='same')
    time_end = time.time()
    print("fftconvolve cost {:.6f} s ".format(time_end-time_start))
    
    # add the area modification
    discrete_area = np_image_plane.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    area_rate = image_area/discrete_area
    
    return conv_fft_result*area_rate

def calcOnImagePlaneBlock(fit_flux,image_coor,image_area,block_image_coor):
    # 计算卷积积分
    np_image_plane = np.zeros((200,200),np.int)
    cv2.fillPoly(np_image_plane, [image_coor], 1)
    
    # add the area modification
    discrete_area = np_image_plane.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    area_rate = image_area/discrete_area
    
    # clear the block
    cv2.fillPoly(np_image_plane, [block_image_coor], 0)
    time_start = time.time()
    conv_fft_result = signal.fftconvolve(np_image_plane, fit_flux/400, mode='same')
    time_end = time.time()
    print("fftconvolve cost {:.6f} s ".format(time_end-time_start))
    
    return conv_fft_result*area_rate


def transformRecevier(np_image,image_coor,receiver_coor,receiver_area,energy_attenuation,shift_pos = False,shift_x = 0,shift_y=0):
    rows,cols = np_image.shape
    # pst1 是Image plane 上的坐标   pst2 s是receiver_plane 上的坐标
    pts1 = np.float32([[image_coor[0][0],image_coor[0][1]],\
                        [image_coor[1][0],image_coor[1][1]],\
                        [image_coor[2][0],image_coor[2][1]]])
    pts2 = np.float32([[receiver_coor[0][0],receiver_coor[0][1]],\
                        [receiver_coor[1][0],receiver_coor[1][1]],\
                        [receiver_coor[2][0],receiver_coor[2][1]]])
    time_start = time.time()
    M = cv2.getAffineTransform(pts1,pts2)
    #print(M)
    np_receiver = cv2.warpAffine(np_image,M,(cols,rows),cv2.INTER_LINEAR)*energy_attenuation
    # 离散的能量修正
    np_image_receiver = np.zeros((200,200),np.int)
    cv2.fillPoly(np_image_receiver, [receiver_coor], 1)
    discrete_area = np_image_receiver.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    area_rate = receiver_area/discrete_area

    if shift_pos:
        pst1_shift = np.float32([[10,10],[10,190],[190,10]])
        pst2_shift = np.float32([[10+shift_x,10+shift_y],[10+shift_x,190+shift_y],[190+shift_x,10+shift_y]])
        M_shift = cv2.getAffineTransform(pst1_shift,pst2_shift)
        np_receiver = cv2.warpAffine(np_receiver,M_shift,(cols,rows),cv2.INTER_LINEAR)
    time_end = time.time()
    print("transform cost {:.6f} s ".format(time_end-time_start))
    return np_receiver*area_rate


def envaluateFlux(gt,res,title="卷积结果"):
    error = res - gt
    gt_max = gt.max()
    res_max = res.max()
    max_error_rate = (res_max-gt_max)/gt_max 
    max_val = max(gt_max,res_max)
    max_val_str = "Maxval: GT {} conv{} error rate {}".format(gt_max,res_max,max_error_rate)
    print(max_val_str)
    
    gt_sum = gt.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    res_sum = res.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    sum_error_rate = (res_sum-gt_sum)/gt_sum
    sum_val_str = "Sum: GT {} conv{} error rate {}".format(gt_sum,res_sum,sum_error_rate)
    print(sum_val_str)
    
    rmse = myUtils.rmse(gt*globalVar.GRID_LEN*globalVar.GRID_LEN,res*globalVar.GRID_LEN*globalVar.GRID_LEN)
    print("RMSE: {}".format(rmse))
    
    if globalVar.IS_PLOT_EVALUATE:
        plt.subplot(131)
        plt.imshow(gt, interpolation='bilinear',origin='lower', \
                       cmap =  cm.jet, vmin=0,vmax=max_val)
        plt.title("ray tracing")
        plt.colorbar()
        
        plt.subplot(132)
        plt.imshow(res, interpolation='bilinear',origin='lower', \
                       cmap =  cm.jet, vmin=0,vmax=max_val)
        plt.title("conv result")
        plt.colorbar()
        
        
        plt.subplot(133)
        plt.imshow(error, interpolation='bilinear',origin='lower', \
                       cmap =  cm.jet, vmin=-20,vmax=20)
        plt.title("error")
        plt.colorbar()
    
    
    