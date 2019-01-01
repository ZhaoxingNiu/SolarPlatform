# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 15:15:11 2019

@author: nzx
"""


import cv2 as cv
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm

import sys
import os
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import myUtils
import globalVar


#设置能够正常显示中文以及负号
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

lim_num = 2
xmin = -1*lim_num
xmax = lim_num
ymin = -1*lim_num
ymax = lim_num

globalVar.FONTSIZE = 24

tick_font_size = 18

def process_sub_file(index,model_name,delim=','):
    flux_spot = np.zeros((200,200))
    for helios_index in range(index,index+1):
        res_path = globalVar.DATA_PATH + "../paper/scene_ps10_flat/{}/equinox_12_#{}.txt".format(model_name,helios_index)
        for sub_index in range(28):
            raytracing_path = globalVar.DATA_PATH + "../paper/scene_ps10_flat/{}/equinox_12_#{}_{}.txt".format(model_name,helios_index,sub_index)
            helios_flux = np.genfromtxt(raytracing_path,delimiter=delim)
            print("equinox_12_#{}_{}:  {}\n".format(helios_index,sub_index,helios_flux.sum()/400))
            flux_spot += helios_flux
        np.savetxt(res_path,flux_spot,fmt='%0.4f',delimiter=delim)

    
def getHeliosTotal(index,model_name,delim=','):
    flux_spot = np.zeros((200,200))
    for helios_index in range(index,index+1):
        raytracing_path = globalVar.DATA_PATH + "../paper/scene_ps10_flat/{}/equinox_12_#{}.txt".format(model_name,helios_index)
        #helios_flux = np.genfromtxt(raytracing_path,delimiter=' ')
        helios_flux = np.genfromtxt(raytracing_path,delimiter= delim)
        flux_spot += helios_flux
    return flux_spot
    
def getHeliosSub(index,model_name,delim=','):
    flux_spot = np.zeros((200,200))
    for helios_index in range(index,index+1):
        for sub_index in range(28):
            raytracing_path = globalVar.DATA_PATH + "../paper/scene_ps10_flat/{}/equinox_12_#{}_{}.txt".format(model_name,helios_index,sub_index)
            #helios_flux = np.genfromtxt(raytracing_path,delimiter=' ')
            helios_flux = np.genfromtxt(raytracing_path,delimiter=delim)
            flux_spot += helios_flux
    return flux_spot


def plot3fig(rt_res, hflcal_res, unizar_res ,conv_res):
    fig = plt.figure(figsize=(15,15))
    
    max_val = max(rt_res.max(), hflcal_res.max(), unizar_res.max(), conv_res.max())
    plot1 = fig.add_subplot(221)
    plot1.imshow(rt_res, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax = max_val)
  
    plot2 = fig.add_subplot(222)
    plot2.imshow(hflcal_res, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax = max_val)
    #plt.colorbar()
    
    plot3 = fig.add_subplot(223)
    plot3.imshow(unizar_res, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax = max_val)
    
    plot4 = fig.add_subplot(224)
    plot4.imshow(conv_res, interpolation='bilinear',origin='lower', \
               cmap = cm.jet, vmin=0,vmax = max_val)
    
def init():
    for index in range(10):
        process_sub_file(index,'raytracing/2048',',')
        #process_sub_file(index,'model_sub_tmp',' ')


def mersure_res(rt_res,conv_res,unizar_res,hflcal_res):
    result = dict()
    
    rt_peak = rt_res.max()
    rt_total = rt_res.sum()/400
        
    conv_peak = conv_res.max()
    conv_total = conv_res.sum()/400
    conv_rmse = myUtils.rmse(conv_res,rt_res)   
    conv_peak_rate = (conv_peak - rt_peak)/rt_peak
    conv_total_rate = (conv_total - rt_total)/rt_total
    result["conv"] = [conv_peak,conv_peak_rate,conv_total,conv_total_rate,conv_rmse]
    
    unizar_peak = unizar_res.max()
    unizar_total = unizar_res.sum()/400
    unizar_rmse = myUtils.rmse(unizar_res,rt_res)   
    unizar_peak_rate = (unizar_peak - rt_peak)/rt_peak
    unizar_total_rate = (unizar_total - rt_total)/rt_total
    result["unizar"] = [unizar_peak,unizar_peak_rate,unizar_total,unizar_total_rate,unizar_rmse]
    
    hflcal_peak = hflcal_res.max()
    hflcal_total = hflcal_res.sum()/400
    hflcal_rmse = myUtils.rmse(hflcal_res,rt_res)   
    hflcal_peak_rate = (hflcal_peak - rt_peak)/rt_peak
    hflcal_total_rate = (hflcal_total - rt_total)/rt_total
    result["hflcal"] = [hflcal_peak,hflcal_peak_rate,hflcal_total,hflcal_total_rate,hflcal_rmse]
    
    return result



def process_the_image_plane(index,delim=','):
    flux_spot = np.zeros((200,200))
    for helios_index in range(index,index+1):
        res_path = globalVar.DATA_PATH + "../imageplane/ps10_sub_image_#{}.txt".format(helios_index)
        for sub_index in range(28):
            raytracing_path = globalVar.DATA_PATH + "../imageplane/ps10_sub_image_{}.txt".format(sub_index)
            helios_flux = np.genfromtxt(raytracing_path,delimiter=delim)
            flux_spot += helios_flux
        np.savetxt(res_path,flux_spot,fmt='%0.4f',delimiter=delim)
#process_the_image_plane(0,delim=',')

if __name__ == "__main__":
    #init()
   
    helios_index = 4
    rt_res = getHeliosTotal(helios_index,'raytracing/2048')
    hflcal_res = getHeliosTotal(helios_index,'hflcal',' ')
    unizar_res = getHeliosTotal(helios_index,'unizar',' ')
    conv_res = getHeliosSub(helios_index,'model_sub_tmp',' ')
    
    
    hflcal_res = np.rot90(hflcal_res,1,(1,0))
    hflcal_res = np.rot90(hflcal_res,1,(1,0))
    hflcal_res = np.rot90(hflcal_res,1,(1,0))
    
    unizar_res = np.rot90(unizar_res,1,(1,0))
    unizar_res = np.rot90(unizar_res,1,(1,0))
    unizar_res = np.rot90(unizar_res,1,(1,0))
    
    
    conv_res = np.rot90(conv_res,1,(1,0))
    conv_res = np.rot90(conv_res,1,(1,0))
    conv_res = np.rot90(conv_res,1,(1,0))
    
    plot3fig(rt_res, hflcal_res, unizar_res ,conv_res)
    mearsurement = mersure_res(rt_res,conv_res,unizar_res,hflcal_res)