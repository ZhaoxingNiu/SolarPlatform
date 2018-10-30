# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:07:32 2018

@author: nzx

@decription: 三次B样条的插值拟合
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep

x = np.linspace(0, 10, 11)
y = np.sin(x)
spl = splrep(x, y)
x2 = np.linspace(0, 10, 200)
y2 = splev(x2, spl)
plt.plot(x, y, 'o', x2, y2)
plt.show()