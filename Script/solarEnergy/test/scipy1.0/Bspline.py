# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:14:05 2018

@author: nzx
"""

# 三次B样条插值
import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

x = np.array([ 0. ,  1.2,  1.9,  3.2,  4. ,  6.5])
y = np.array([ 0. ,  2.3,  3. ,  4.3,  2.9,  3.1])

t, c, k = interpolate.splrep(x, y, s=0, k=4)
print('''\
t: {}
c: {}
k: {}
'''.format(t, c, k))
N = 100
xmin, xmax = x.min(), x.max()
xx = np.linspace(xmin, xmax, N)
spline = interpolate.BSpline(t, c  , k, extrapolate=False)

plt.plot(x, y, 'bo', label='Original points')
plt.plot(xx, spline(xx), 'r', label='BSpline')
plt.grid()
plt.legend(loc='best')
plt.show()


# 多项式拟合
x = np.arange(1, 17, 1)
y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
z1 = np.polyfit(x, y, 4)#3为多项式最高次幂，结果为多项式的各个系数
#最高次幂3，得到4个系数,从高次到低次排列
#最高次幂取几要视情况而定
p1 = np.poly1d(z1)#将系数代入方程，得到函式p1
print(p1)#多项式方程

x1=np.linspace(x.min(),x.max(),100)#x给定数据太少，方程曲线不光滑，多取x值得到光滑曲线
pp1=p1(x1)#x1代入多项式，得到pp1,代入matplotlib中画多项式曲线
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.scatter(x,y,color='g',label='散点图')#x，y散点图
plt.plot(x,y,color='r',label='连线图')#x,y线形图
plt.plot(x1,pp1,color='b',label='拟合图')#100个x及对应y值绘制的曲线
#可应用于各个行业的数值预估
plt.legend(loc='best')
plt.show()