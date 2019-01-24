# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 21:02:35 2019

@author: nzx
"""


import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

plt.rcParams['font.sans-serif']=['SimHei']#显示中文

#x = np.arange(10)
#y = np.sin(x)
x = np.array([0,1,2,3,4,5,6,7,8])
y = np.array([0,1.0,8.0,27,64,125,216,343,512])

cs = UnivariateSpline(x, y, s=0)
xs = np.arange(-0.5, 8, 0.1)
plt.figure(figsize=(6.5, 4))
plt.plot(x, y, 'o', label='data')
plt.plot(xs, xs*xs*xs , label='real')
plt.plot(xs, cs(xs), label="S")
plt.legend()


test_x = np.array([1.5, 2.5, 3.5])
test_y = cs(test_x)