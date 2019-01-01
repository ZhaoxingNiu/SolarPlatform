# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 13:34:54 2018

@author: nzx
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


flux_list = []

def process_raytracing(index):
    flux_spot = np.zeros((200,200))
    for helios_index in range(index):
        res_path = globalVar.DATA_PATH + "../paper/scene_ps10_flat/raytracing/2048/equinox_12_#{}.txt".format(helios_index)
        for sub_index in range(28):
            raytracing_path = globalVar.DATA_PATH + "../paper/scene_ps10_flat/raytracing/2048/equinox_12_#{}_{}.txt".format(helios_index,sub_index)
            helios_flux = np.genfromtxt(raytracing_path,delimiter=',')
            flux_spot += helios_flux
        np.savetxt(res_path,flux_spot,fmt='%0.4f',delimiter=',')



def showHeliosRT(index):
    flux_spot = np.zeros((200,200))
    for helios_index in range(index,index+1):
        for sub_index in range(28):
            #raytracing_path = globalVar.DATA_PATH + "../paper/scene1/raytracing/1024/equinox_12_#{}.txt".format(helios_index)
            raytracing_path = globalVar.DATA_PATH + "../paper/scene_ps10_flat/raytracing/2048/equinox_12_#{}_{}.txt".format(helios_index,sub_index)
            #helios_flux = np.genfromtxt(raytracing_path,delimiter=' ')
            helios_flux = np.genfromtxt(raytracing_path,delimiter=',')
            flux_spot += helios_flux
            flux_list.append(helios_flux)
    
    plt.imshow(flux_spot, interpolation='bilinear',origin='lower', \
                   cmap = cm.jet, vmin=0)
    #plt.colorbar()
    return flux_spot
    

def showHelioConv(index):
    conv_path = globalVar.DATA_PATH + "../paper/scene_ps10_flat/model/equinox_12_#{}.txt".format(index)
    helios_flux = np.genfromtxt(conv_path,delimiter=' ')
    
    helios_flux = np.rot90(helios_flux,1,(1,0))
    helios_flux = np.rot90(helios_flux,1,(1,0))
    helios_flux = np.rot90(helios_flux,1,(1,0))

    plt.imshow(helios_flux, interpolation='bilinear',origin='lower', \
                   cmap = cm.jet, vmin=0)
    #plt.colorbar()
    return helios_flux

def showHeliosHFLCAL(index):
    conv_path = globalVar.DATA_PATH + "../paper/scene_ps10_flat/hflcal/equinox_12_#{}.txt".format(index)
    helios_flux = np.genfromtxt(conv_path,delimiter=' ')
    
    helios_flux = np.rot90(helios_flux,1,(1,0))
    helios_flux = np.rot90(helios_flux,1,(1,0))
    helios_flux = np.rot90(helios_flux,1,(1,0))

    plt.imshow(helios_flux, interpolation='bilinear',origin='lower', \
                   cmap = cm.jet, vmin=0)
    #plt.colorbar()
    return helios_flux


def showHeliosSubRT(index,sub):
    flux_spot = np.zeros((200,200))
    for helios_index in range(index,index+1):
        for sub_index in range(sub,sub+1):
            #raytracing_path = globalVar.DATA_PATH + "../paper/scene1/raytracing/1024/equinox_12_#{}.txt".format(helios_index)
            raytracing_path = globalVar.DATA_PATH + "../paper/scene_ps10_flat/raytracing/2048/equinox_12_#{}_{}.txt".format(helios_index,sub_index)
            #helios_flux = np.genfromtxt(raytracing_path,delimiter=' ')
            helios_flux = np.genfromtxt(raytracing_path,delimiter=',')
            flux_spot += helios_flux
            flux_list.append(helios_flux)
    plt.imshow(flux_spot, interpolation='bilinear',origin='lower', \
                   cmap = cm.jet, vmin=0)
    plt.colorbar()
    
    

if __name__ == '__main__':
    # 这个函数的功能是将28个子文件合成一个
    #process_raytracing(50)
    helio_index = 5
    flux_spot_rt = showHeliosRT(helio_index)
    #flux_spot_hflcal = showHeliosHFLCAL(helio_index)
    
    #flux_spot_hflcal = showHeliosHFLCAL(helio_index)
    
    #flux_spot_conv = showHelioConv(0)
    