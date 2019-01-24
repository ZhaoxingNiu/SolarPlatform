# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:58:33 2018

@author: nzx
"""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']#显示中文

#x = np.arange(10)
#y = np.sin(x)
x = np.array([0,1,2,3,4,5,6,7,8])
y = np.array([0,1.0,8.0,27,64,125,216,343,512])

cs = CubicSpline(x, y)
xs = np.arange(-0.5, 8, 0.1)
plt.figure(figsize=(6.5, 4))
plt.plot(x, y, 'o', label='data')
plt.plot(xs, cs(xs), label="S")
plt.plot(xs, cs(xs, 1), label="S'")
plt.xlim(-0.5, 9.5)
#plt.legend(loc='lower left', ncol=2)


test_x = np.array([1.5, 2.5, 3.5])
test_y = cs(test_x, 1)