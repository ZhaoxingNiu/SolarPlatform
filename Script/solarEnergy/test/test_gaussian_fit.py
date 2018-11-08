# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:08:33 2018

@author: nzx
"""


import numpy as np
import pylab as plt
#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import math 

x = ar(range(10))
y = ar([0,1,2,3,4,5,4,3,2,1])
 
 
def gaussian(x, a, miu, sigma):
    return a/(math.sqrt(math.pi*2)*sigma)*np.exp(-(x-miu)*(x-miu)/2/sigma/sigma)
 
 
popt,pcov = curve_fit(gaussian,x,y)
print(popt)
print(pcov)
 
plt.plot(x,y,'b+:',label='data')
plt.plot(x,gaussian(x,*popt),'ro:',label='fit')
plt.legend()
plt.show()