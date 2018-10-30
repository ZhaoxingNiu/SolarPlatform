# -*- coding: utf-8 -*-
"""
Created on Thu May  3 00:40:36 2018

@author: nzx
"""

import sys 
sys.path.append("../") 

import numpy as np

import math
import matplotlib.pyplot as plt
import myUtils
import globalVar
import CDF
import PDF


#import sys
#sys.path.append('./projection')
#import imageplane
#import calcShape
#from vector import Vector


#设置能够正常显示中文以及负号
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


globalVar.IS_PLOT_CDF = False

def differentCDF(step_r,fit_fun0,fit_fun225,fit_fun30,fit_fun45):
    fig1 = plt.figure('能量累计分布')
    plt.subplot(121)
    plt.plot(step_r,fit_fun0(step_r),'k',lw =3,label= '0')
    plt.plot(step_r,fit_fun225(step_r),'b',lw =3,label= '22.5')
    plt.plot(step_r,fit_fun30(step_r),'c',lw =3,label= '30')
    plt.plot(step_r,fit_fun45(step_r),'g',lw =3,label= '45')
    plt.legend()
    plt.title("能量分布CDF")
    
    plt.subplot(122)
    plt.plot(step_r,fit_fun0(step_r)/math.cos(0*math.pi/180),'k',lw =3,label= '0')
    plt.plot(step_r,fit_fun225(step_r)/math.cos(22.5*math.pi/180),'b',lw =3,label= '22.5')
    plt.plot(step_r,fit_fun30(step_r)/math.cos(30*math.pi/180),'c',lw =3,label= '30')
    plt.plot(step_r,fit_fun45(step_r)/math.cos(45*math.pi/180),'g',lw =3,label= '45')
    plt.legend()
    plt.title("能量分布CDF除以余弦值")
    
def differentCDF_1(step_r,fit_fun0,fit_fun225,fit_fun30,fit_fun45):
    fig2 = plt.figure('能量导数分布')
    plt.subplot(121)
    plt.plot(step_r,fit_fun0(step_r,1),'k',lw =3,label= '0')
    plt.plot(step_r,fit_fun225(step_r,1),'b',lw =3,label= '22.5')
    plt.plot(step_r,fit_fun30(step_r,1),'c',lw =3,label= '30')
    plt.plot(step_r,fit_fun45(step_r,1),'g',lw =3,label= '45')
    plt.legend()
    plt.title("能量分布CDF导数")
    
    plt.subplot(122)
    plt.plot(step_r,fit_fun0(step_r,1)/math.cos(0*math.pi/180),'k',lw =3,label= '0')
    plt.plot(step_r,fit_fun225(step_r,1)/math.cos(22.5*math.pi/180),'b',lw =3,label= '22.5')
    plt.plot(step_r,fit_fun30(step_r,1)/math.cos(30*math.pi/180),'c',lw =3,label= '30')
    plt.plot(step_r,fit_fun45(step_r,1)/math.cos(45*math.pi/180),'g',lw =3,label= '45')
    plt.legend()
    plt.title("能量分布CDF导数除以余弦值")
    
def differentEnergy(step_r,fit_fun0,fit_fun225,fit_fun30,fit_fun45):
    def mod_fit_fun(func,r):
        if(r<=globalVar.DISTANCE_THRESHOLD):
            result = func(globalVar.DISTANCE_THRESHOLD)/4/globalVar.DISTANCE_THRESHOLD/globalVar.DISTANCE_THRESHOLD
        else:
            result = func(r,1)/math.pi/2/r
        return result
    
    fig2 = plt.figure('能量分布')
    plt.subplot(121)
    
    flux_line0    = [mod_fit_fun(fit_fun0,r) for r in step_r]
    flux_line225  = [mod_fit_fun(fit_fun225,r) for r in step_r]
    flux_line30   = [mod_fit_fun(fit_fun30,r) for r in step_r]
    flux_line45   = [mod_fit_fun(fit_fun45,r) for r in step_r]
    
    
    plt.plot(step_r,flux_line0,'k',lw =3,label= '0')
    plt.plot(step_r,flux_line225,'b',lw =3,label= '22.5')
    plt.plot(step_r,flux_line30,'c',lw =3,label= '30')
    plt.plot(step_r,flux_line45,'g',lw =3,label= '45')
    plt.legend()
    plt.title("能量分布")
    
    flux_line0    = [mod_fit_fun(fit_fun0,r)/math.cos(0*math.pi/180) for r in step_r]
    flux_line225  = [mod_fit_fun(fit_fun225,r)/math.cos(22.5*math.pi/180) for r in step_r]
    flux_line30   = [mod_fit_fun(fit_fun30,r)/math.cos(30*math.pi/180) for r in step_r]
    flux_line45   = [mod_fit_fun(fit_fun45,r)/math.cos(45*math.pi/180) for r in step_r]
    
    plt.subplot(122)
    plt.plot(step_r,flux_line0,'k',lw =3,label= '0')
    plt.plot(step_r,flux_line225,'b',lw =3,label= '22.5')
    plt.plot(step_r,flux_line30,'c',lw =3,label= '30')
    plt.plot(step_r,flux_line45,'g',lw =3,label= '45')
    plt.legend()
    plt.title("能量分布除以余弦值")
    
def differentTrue(step_r,energy0,energy225,energy30,energy45):
    fig2 = plt.figure('统计能量分布')
    plt.subplot(121)
    plt.plot(step_r,energy0,'k',lw =3,label= '0')
    plt.plot(step_r,energy225,'b',lw =3,label= '22.5')
    plt.plot(step_r,energy30,'c',lw =3,label= '30')
    plt.plot(step_r,energy45,'g',lw =3,label= '45')
    plt.legend()
    plt.title("Ray tracing PDF")
    
    plt.subplot(122)
    energy0 = [e /math.cos(0*math.pi/180) for e in energy0]
    energy225 = [e /math.cos(22.5*math.pi/180) for e in energy225]
    energy30 = [e /math.cos(30*math.pi/180) for e in energy30]
    energy45 = [e /math.cos(45*math.pi/180) for e in energy45]
    
    plt.plot(step_r,energy0,'k',lw =3,label= '0')
    plt.plot(step_r,energy225,'b',lw =3,label= '22.5')
    plt.plot(step_r,energy30,'c',lw =3,label= '30')
    plt.plot(step_r,energy45,'g',lw =3,label= '45')
    
    plt.legend()
    plt.title("Convolution PDF")
    

def differentPDF(pdf0,pdf225,pdf30,pdf45):
    fig2 = plt.figure('PDF能量比较')
    pdf0_transform = pdf0*math.cos(45*math.pi/180)
    max_val = math.ceil(pdf0_transform.max()/50)*50
    plt.subplot(131)
    plt.imshow(pdf0_transform, interpolation='bilinear',origin='lower', \
                   cmap =  plt.cm.jet, vmin=0,vmax=max_val)
    
    plt.colorbar()
    plt.title("angle0 PDF transform")
    
    plt.subplot(132)
    plt.imshow(pdf45, interpolation='bilinear',origin='lower', \
                   cmap =  plt.cm.jet, vmin=0,vmax=max_val)
    
    plt.colorbar()
    plt.title("angle45 PDF")
    
    error = pdf0_transform - pdf45
    plt.subplot(133)
    plt.imshow(error, interpolation='bilinear',origin='lower', \
                   cmap =  plt.cm.jet, vmin=-20,vmax=20)
    
    plt.colorbar()
    plt.title("error")
    
    fit_rmse = myUtils.rmse(pdf0_transform,pdf45)
    total_energy = pdf45.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    fit_total_energy = pdf45.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    energy_rate = (fit_total_energy-total_energy)/total_energy*100
    print("PDF RMSE: {} angle 0 transform: {} angle 45 : {}  error： {:.2f}%".format(fit_rmse,total_energy,fit_total_energy,energy_rate))  
    

def differentPDFTrue(pdf0,pdf1,angle):
    fig2 = plt.figure('Raytracing PDF能量比较')
    pdf0_transform = pdf0*math.cos(67.5*math.pi/180)
    max_val = math.ceil(pdf0_transform.max()/50)*50
    plt.subplot(131)
    plt.imshow(pdf0_transform, interpolation='bilinear',origin='lower', \
                   cmap =  plt.cm.jet, vmin=0,vmax=max_val)
    
    plt.colorbar()
    plt.title("angle0 PDF transform")
    
    plt.subplot(132)
    plt.imshow(pdf1, interpolation='bilinear',origin='lower', \
                   cmap =  plt.cm.jet, vmin=0,vmax=max_val)
    
    plt.colorbar()
    plt.title("angle" + str(angle)+ "  PDF")
    
    error = pdf0_transform - pdf1
    plt.subplot(133)
    plt.imshow(error, interpolation='bilinear',origin='lower', \
                   cmap =  plt.cm.jet, vmin=-20,vmax=20)
    
    plt.colorbar()
    plt.title("error")
    
    fit_rmse = myUtils.rmse(pdf0_transform,pdf1)
    total_energy = pdf1.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    fit_total_energy = pdf0_transform.sum()*globalVar.GRID_LEN*globalVar.GRID_LEN
    energy_rate = (fit_total_energy-total_energy)/total_energy*100
    print("PDF RMSE: {} angle 0 transform: {} angle : {}  error： {:.2f}%".format(fit_rmse,fit_total_energy,total_energy,energy_rate))  
    
         
if __name__ == '__main__':
    process_distance = 500
    print('...................')
    print('load the angle0...')
    onepoint_path = globalVar.DATA_PATH + "incident_angle/onepoint_odd_distance005_0_500.txt"
    onepoint_flux = np.genfromtxt(onepoint_path,delimiter=',')
    onepoint_flux0 = myUtils.smoothData(onepoint_flux) #*math.sqrt(2)
    
    # 统计拟合得到CDF
    print('count the flux data...')
    energy_static0,energy_static_per0,step_r = CDF.staticFlux(onepoint_flux0,width=globalVar.RECE_WIDTH,height = globalVar.RECE_HEIGHT)
    energy_accu0 = CDF.accumulateFlux(energy_static0)
    fit_fun0 = CDF.getCDF(step_r,energy_accu0,energy_static_per0,1,'angle 0')

    # 通过对应的 CDF去计算 PDF
    print("calculate the RMSE")
    fit_flux0 = PDF.getPDFFlux(fit_fun0, width=globalVar.RECE_WIDTH, height = globalVar.RECE_HEIGHT)
    PDF.envaluatePDF(onepoint_flux0,fit_flux0)
    
    
    print('...................')
    print('load the angle225...')
    onepoint_path = globalVar.DATA_PATH + "incident_angle/onepoint_odd_distance005_45_500.txt"
    onepoint_flux = np.genfromtxt(onepoint_path,delimiter=',')
    onepoint_flux225 = myUtils.smoothData(onepoint_flux) #*math.sqrt(2) 
    # 统计拟合得到CDF
    print('count the flux data...')
    energy_static225,energy_static_per225,step_r = CDF.staticFlux(onepoint_flux225,width=globalVar.RECE_WIDTH,height = globalVar.RECE_HEIGHT)
    energy_accu225 = CDF.accumulateFlux(energy_static225)
    fit_fun225 = CDF.getCDF(step_r,energy_accu225,energy_static_per225,1,'angle 22.5')
    # 通过对应的 CDF去计算 PDF
    print("calculate the RMSE")
    fit_flux225 = PDF.getPDFFlux(fit_fun225, width=globalVar.RECE_WIDTH, height = globalVar.RECE_HEIGHT)
    PDF.envaluatePDF(onepoint_flux225,fit_flux225)
    
    
    print('...................')
    print('load the angle30...')
    onepoint_path = globalVar.DATA_PATH + "incident_angle/onepoint_odd_distance005_60_500.txt"
    onepoint_flux = np.genfromtxt(onepoint_path,delimiter=',')
    onepoint_flux30 = myUtils.smoothData(onepoint_flux) #*math.sqrt(2)
    # 统计拟合得到CDF
    print('count the flux data...')
    energy_static30,energy_static_per30,step_r = CDF.staticFlux(onepoint_flux30,width=globalVar.RECE_WIDTH,height = globalVar.RECE_HEIGHT)
    energy_accu30 = CDF.accumulateFlux(energy_static30)
    fit_fun30 = CDF.getCDF(step_r,energy_accu30,energy_static_per30,1,'angle 30')
    # 通过对应的 CDF去计算 PDF
    print("calculate the RMSE")
    fit_flux30 = PDF.getPDFFlux(fit_fun30, width=globalVar.RECE_WIDTH, height = globalVar.RECE_HEIGHT)
    PDF.envaluatePDF(onepoint_flux30,fit_flux30)
        
    
    print('...................')
    print('load the angle45...')
    onepoint_path = globalVar.DATA_PATH + "incident_angle/onepoint_odd_distance005_90_500.txt"
    onepoint_flux = np.genfromtxt(onepoint_path,delimiter=',')
    onepoint_flux45 = myUtils.smoothData(onepoint_flux) #*math.sqrt(2)
    # 统计拟合得到CDF
    print('count the flux data...')
    energy_static45,energy_static_per45,step_r = CDF.staticFlux(onepoint_flux45,width=globalVar.RECE_WIDTH,height = globalVar.RECE_HEIGHT)
    energy_accu45 = CDF.accumulateFlux(energy_static45)
    fit_fun45 = CDF.getCDF(step_r,energy_accu45,energy_static_per45,1,'angle 45')
    # 通过对应的 CDF去计算 PDF
    print("calculate the RMSE")
    fit_flux45 = PDF.getPDFFlux(fit_fun45, width=globalVar.RECE_WIDTH, height = globalVar.RECE_HEIGHT)
    PDF.envaluatePDF(onepoint_flux45,fit_flux45)
    
    print('...................')
    print('load the angle725...')
    onepoint_path = globalVar.DATA_PATH + "incident_angle/onepoint_odd_distance005_145_500.txt"
    onepoint_flux = np.genfromtxt(onepoint_path,delimiter=',')
    onepoint_flux725 = myUtils.smoothData(onepoint_flux) #*math.sqrt(2)
    
    # 统计拟合得到CDF
    print('count the flux data...')
    energy_static725,energy_static_per725,step_r = CDF.staticFlux(onepoint_flux725,width=globalVar.RECE_WIDTH,height = globalVar.RECE_HEIGHT)
    energy_accu725 = CDF.accumulateFlux(energy_static725)
    fit_fun725 = CDF.getCDF(step_r,energy_accu725,energy_static_per725,1,'angle 725')

    # 通过对应的 CDF去计算 PDF
    print("calculate the RMSE")
    fit_flux725 = PDF.getPDFFlux(fit_fun725, width=globalVar.RECE_WIDTH, height = globalVar.RECE_HEIGHT)
    PDF.envaluatePDF(onepoint_flux725,fit_flux725)
    
        
    ##绘图
    #differentCDF(step_r,fit_fun0,fit_fun225,fit_fun30,fit_fun45)
    #differentCDF_1(step_r,fit_fun0,fit_fun225,fit_fun30,fit_fun45)
    #differentEnergy(step_r,fit_fun0,fit_fun225,fit_fun30,fit_fun45)
    #differentTrue(step_r,energy_static_per0,energy_static_per225,energy_static_per30,energy_static_per45)
    #differentPDF(fit_flux0,fit_flux225,fit_flux30,fit_flux45)
    differentPDFTrue(onepoint_flux0,onepoint_flux725,725)