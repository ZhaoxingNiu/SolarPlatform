# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:03:12 2017

@author: nzx
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm 

def showResultPic(filepath,totalEnergy = 2640 ,pixel_num=20):
    fig = plt.figure()
    print("filepath %s" % (filepath))
    print("totalEnergy %f" % (totalEnergy))
    print("pixel_num %d  per meter"% (pixel_num))

    if(os.path.isfile(filepath)):
        fig = plt.figure()
        result= np.loadtxt(filepath,delimiter=',')
        maxVal = np.max(result)
        sumVal = np.sum(result)/(pixel_num*pixel_num)

        tmpResult = result.copy()
        tmpResult = tmpResult.reshape(tmpResult.shape[0]*tmpResult.shape[1])
        tmpResult.sort()

        maxVal_999 = tmpResult[int(0.999*len(tmpResult))]
        reveiceRate = 100*sumVal/totalEnergy

        colorbarMaxVal = 100
        if(maxVal_999 >=colorbarMaxVal):
            colorbarMaxVal = maxVal_999
        im = plt.imshow(result, interpolation='bilinear',origin='lower', cmap =  cm.jet, vmin=0, vmax=colorbarMaxVal) 
        #设置title  
        titleStr='$ maxVal$='+str(maxVal)+' $maxVal999$=' + str(maxVal_999)  +'\n $ sumEnergy$='+str(float("%.2f" % sumVal)) + ' $ reveiceRate$='+str(float("%.2f" % reveiceRate))+ " %"
        plt.title(titleStr)
        plt.colorbar(shrink=0.8)
        plt.show()

        savePath = "./"
        if(~os.path.isdir(savePath) == 1):
            os.makedirs(savePath)
        fileBaseName = os.path.splitext(os.path.basename(filepath))[0]
        resultFilename = savePath+fileBaseName+".png"
        print(resultFilename)
        fig.savefig(resultFilename,dpi= 300)
    else:
        print("File %s: not exist!!!!" % (filepath)) 
    

if __name__ == '__main__':
    for distance_rate in range(1,2):
        filepath = "../../SimulResult/onepoint/one_point_angel0_distance_500.txt"
        totalEnergy = 880
        pixel_num = 20
        showResultPic(filepath,totalEnergy,pixel_num)

    