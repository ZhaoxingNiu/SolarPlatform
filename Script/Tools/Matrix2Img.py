# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:45:41 2017

@author: nzx
"""


import numpy as np
from PIL import Image
from pylab import uint8
import matplotlib.pyplot as plt
#import matplotlib as plts

def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()  
    width,height = im.size
    im = im.convert("L") 
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data


def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


if __name__ == '__main__':
    img = np.loadtxt('./face2face_shadow-1.txt',delimiter=',')
    minVal = np.max(img)
    maxVal = np.max(img)
    meanVal = np.mean(img)
    sumVal = np.sum(img);
    
                   
    imgShow = MatrixToImage(img/maxVal);
    imgShow.show();
    imgShow.save('result.jpg');