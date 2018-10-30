# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:55:33 2018

@author: nzx
"""

import numpy as np
import math
import matplotlib.pyplot as plt
#from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate

plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

## 多项式拟合
#x = np.arange(1, 17, 1)
#y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
#z1 = np.polyfit(x, y, 4)#3为多项式最高次幂，结果为多项式的各个系数
##最高次幂3，得到4个系数,从高次到低次排列
##最高次幂取几要视情况而定
#p1 = np.poly1d(z1)#将系数代入方程，得到函式p1
#print(p1)#多项式方程
#
#x1=np.linspace(x.min(),x.max(),100)#x给定数据太少，方程曲线不光滑，多取x值得到光滑曲线
#pp1=p1(x1)#x1代入多项式，得到pp1,代入matplotlib中画多项式曲线
#plt.rcParams['font.sans-serif']=['SimHei']#显示中文
#plt.scatter(x,y,color='g',label='散点图')#x，y散点图
#plt.plot(x,y,color='r',label='连线图')#x,y线形图
#plt.plot(x1,pp1,color='b',label='拟合图')#100个x及对应y值绘制的曲线
##可应用于各个行业的数值预估
#plt.legend(loc='best')
#plt.show()


#x = np.linspace(-3, 3, 50)
#y = np.exp(-x**2) + 0.1 * np.random.randn(50)
#plt.plot(x, y, 'ro', ms=5)
#
#
#spl = UnivariateSpline(x, y,k=5)
#xs = np.linspace(-3, 3, 1000)
#plt.plot(xs, spl(xs), 'g', lw=3)
#
#spl.set_smoothing_factor(0.5)
#plt.plot(xs, spl(xs), 'b', lw=3)


#Radius = 5
#
#def half_circle(x):
#    return (Radius**2-x**2)**0.5
#
#def half_sphere(x,y):
#    return (Radius**2-x**2-y**2)**0.2
#
#
#def half_rect(x):
#    return 4
#result,err = integrate.dblquad(half_sphere,-1,1,
#                  lambda x: -half_circle(x),
#                  lambda x: half_circle(x))
    

#def test_core(x,y):
#    pos_r = (x**2+y**2)*0.5
#    return fit_fun(pos_r,1)/2/math.pi/pos_r
#
#result,err = integrate.dblquad(test_core,-4,4,
#                  lambda x: -half_rect(x),
#                  lambda x: half_rect(x))






