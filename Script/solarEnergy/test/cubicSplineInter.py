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
x = np.array([0,1,2,3,4,5])
y = np.array([0,1.4,2.5,9.34,12.33,22.33])

cs = CubicSpline(x, y)
xs = np.arange(-0.5, 6, 0.1)
plt.figure(figsize=(6.5, 4))
plt.plot(x, y, 'o', label='data')
plt.plot(xs, np.sin(xs), label='true')
plt.plot(xs, cs(xs), label="S")
plt.plot(xs, cs(xs, 1), label="S'")
plt.plot(xs, cs(xs, 2), label="S''")
plt.plot(xs, cs(xs, 3), label="S'''")
plt.xlim(-0.5, 9.5)
plt.legend(loc='lower left', ncol=2)


