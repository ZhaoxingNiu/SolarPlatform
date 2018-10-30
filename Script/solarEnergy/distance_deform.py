# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:14:05 2018

@author: nzx

这个函数的主要作用是来验证一下 不同距离的CDF之间的 可变性 
比较不同距离 变换得到的  CDF 以及 PDF  的差异性

"""


import math
import numpy as np
import time
#import pickle
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm 



#设置能够正常显示中文以及负号
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 指定数据的位置以及输出的位置
DATA_PATH = "C:/Users/Administrator/Desktop/data/"
# 统计能量分布时 R 的bin的大小
STEP_R = 0.05
# GRID_LEN  对应的是接收面的格子边长
GRID_LEN = 0.05

# 不适用Threshold阈值的结果
DISTANCE_THRESHOLD = 0.1

# 接收面的长  接收面的宽
RECE_WIDTH = 10.05
RECE_HEIGHT = 10.05


#==============================================================================
# staticFlux 统计光斑的分布
# width,height  表示的是接收面的实际尺寸

# 返回能量的积分和对应的R
#==============================================================================
def staticFlux(data,width = 10.05 ,height = 10.05):
    grid_x, grid_y = data.shape
    step_x = width / grid_x
    step_y = height/ grid_y
    grid_size = step_x*step_y
    assert abs(step_x - step_y) < 0.01
    
    # 统计能量的值
    step_r = STEP_R
    # 统计边长0.71 部分的能量就可以了，sqrt(2)/2
    energy_static  = np.zeros(int(max(width,height) * 0.75 / step_r))
    energy_static_count  = np.zeros(int(max(width,height) * 0.75 / step_r))
    energy_static_per  = np.zeros(int(max(width,height) * 0.75 / step_r))
    energy_dis = [ (i+0.5)*step_r for i in range(energy_static.size)]
    
    for x in range(grid_x):
        for y in range(grid_y):
            pos_x = -width/2 + (x+0.5)*step_x
            pox_y = -height/2 + (y+0.5)*step_y
            pos_r = (pos_x**2 + pox_y **2 )**0.5
            energy_static[math.floor(pos_r/step_r)] += data[x,y]*grid_size
            energy_static_count[math.floor(pos_r/step_r)] += 1
            
    for i in range(energy_static.size):
        if(energy_static_count[i]!=0):    
            energy_static_per[i] = energy_static[i]/energy_static_count[i]/(GRID_LEN*GRID_LEN)
    return energy_static,energy_static_per,energy_dis
               
def staticFluxFun(func ,width = 10.05 ,height = 10.05):
    # 统计能量的值
    step_r = STEP_R
    # 统计边长0.71 部分的能量就可以了，sqrt(2)/2
    energy_static  = np.zeros(int(max(width,height) * 0.75 / step_r))
    energy_static_count  = np.zeros(int(max(width,height) * 0.75 / step_r))
    energy_static_per  = np.zeros(int(max(width,height) * 0.75 / step_r))
    energy_dis = [ (i+0.5)*step_r for i in range(energy_static.size)]
    
    step_x = step_y = 0.01
    grid_size = step_x*step_y
    grid_x = int(width/step_x)
    grid_y = int(height/step_y)
    
    for x in range(grid_x):
        for y in range(grid_y):
            pos_x = -width/2 + (x+0.5)*step_x
            pos_y = -height/2 + (y+0.5)*step_y
            pos_r = (pos_x**2 + pos_y **2 )**0.5
            flux_val =  max(func(pos_x,pos_y),0)
            if pos_r>5:
                flux_val = 0
            energy_static[math.floor(pos_r/step_r)] += flux_val*grid_size
            energy_static_count[math.floor(pos_r/step_r)] += 1
            
    for i in range(energy_static.size):
        if(energy_static_count[i]!=0):    
            energy_static_per[i] = energy_static[i]/energy_static_count[i]/grid_size
    return energy_static,energy_static_per,energy_dis

#==============================================================================
#  计算统计能量分布，统计r内分布的总能量
#==============================================================================
def accumulateFlux(energy_static):
    energy_accu = np.zeros(energy_static.size)
    energy_sum = 0
    for i in range(energy_static.size):
        energy_sum +=  energy_static[i]
        energy_accu[i] = energy_sum
    return energy_accu

#==============================================================================
# 这个函数的主要目的是计算 RMSE
#==============================================================================
def rmse(predictions,targets):
    assert predictions.size == targets.size
    return np.sqrt(np.mean((predictions-targets)**2))
    
# =============================================================================
# 这个函数的主要目的就是计算得到CDF函数，然后进行求解
# fit_type  1  带权重的B样条拟合   2 B样条拟合  3 B样条插值
# =============================================================================
# 使用与scipy 0.19 拟合
from scipy.interpolate import UnivariateSpline
# 插值
from scipy.interpolate import CubicSpline
def getCDF(step_r,energy_accu,energy_static_per,fit_type =1):
    is_plot = False
    # 对于 距离进行抽样 得到对应的数据
    sample_step = int(1/STEP_R/2)  #大约半米左右取一个点
    sample_index = [ x for x in range(0,energy_accu.size,sample_step) ]
    sample_r = [step_r[x] for x in sample_index]
    sample_accu = [ energy_accu[x] for x in sample_index]
    sample_r.insert(0,0)
    sample_accu.insert(0,0)  
    # # 使用样条函数函数进行拟合 权重可以选也可以不选
    if fit_type == 1:
        step_r_weight = [1/x for x in step_r]
        fit_fun = UnivariateSpline(step_r, energy_accu, w = step_r_weight)
        plot_title = 'Bspline fitting(Weighted)'
    elif fit_type == 2:
        # fit_fun.set_smoothing_factor(100)
        fit_fun = UnivariateSpline(sample_r, sample_accu)
        plot_title = 'Bspline fitting'
    elif fit_type == 3:
        plt.plot(sample_r,sample_accu,'ro',ms=5)
        fit_fun = CubicSpline(sample_r, sample_accu)
        plot_title = 'Bspline interplote'
    def mod_fit_fun(r):
        if(r<=DISTANCE_THRESHOLD):
            result = fit_fun(DISTANCE_THRESHOLD)/4/DISTANCE_THRESHOLD/DISTANCE_THRESHOLD
        else:
            result = fit_fun(r,1)/math.pi/2/r
        return result

    if is_plot:
        fig1 = plt.figure('fig_cdf')
        plt.plot(step_r,energy_static_per,'k',label='true flux',lw=3)
        plt.plot(step_r,fit_fun(step_r),'b',lw =3,label= plot_title)
        plt.plot(step_r,fit_fun(step_r,1),'c',lw =3,label=plot_title+' \'')       
        # flux_line = fit_fun(step_r,1)/math.pi/2/step_r
        flux_line = [mod_fit_fun(r) for r in step_r]
        plt.plot(step_r,flux_line,'g',lw =3,label='flux fit ')    
        plt.grid(True)
        plt.title(plot_title)
        plt.xlabel(r'$d_{receiver}(m)$')
        plt.ylabel(r'$Energy(W)\ |\ Energy\ densify(W/{m^2})$')
        plt.legend(loc='upper left')
        fig1.savefig(plot_title + ".png",dpi=400)
        #plt.show()
        plt.xlim(0,8)
        #plt.ylim(0,900)
    return fit_fun

def getPDFFlux(function,width = 10.05,height = 10.05):
    grid_x = round(width/GRID_LEN)
    grid_y = round(height/GRID_LEN)
    fit_flux = np.zeros((grid_x,grid_y))
    
    compute_threshold = DISTANCE_THRESHOLD
    for x in range(grid_x):
        for y in range(grid_y):
            pos_x = -width/2 + (x+0.5)*GRID_LEN
            pos_y = -height/2 + (y+0.5)*GRID_LEN
            pos_r = (pos_x**2 + pos_y **2 )**0.5
            #计算flux_map
            if pos_r <=compute_threshold:
                fit_flux[x,y] = function(pos_r)/math.pi/compute_threshold/compute_threshold   #  math.pi  还是4
            else:
                fit_flux[x,y] = function(pos_r,1)/2/math.pi/pos_r
    # 修正小于0 的部分以及峰值
    fit_flux[fit_flux<0] = 0
    return fit_flux

def getPDFFluxTransform(fit_func,func_distance,predict_diatance,width = 10.05,height = 10.05):
    grid_x = round(width/GRID_LEN)
    grid_y = round(height/GRID_LEN)
    fit_flux = np.zeros((grid_x,grid_y))
    
    compute_threshold = DISTANCE_THRESHOLD
    distance_rate = func_distance/predict_diatance
    #由于能量衰减的原因，需要整体变换一下
    air_rate1 = AirAttenuation(func_distance)
    air_rate2 = AirAttenuation(predict_diatance)
    attenuation_rate = air_rate2/air_rate1
    
    for x in range(grid_x):
        for y in range(grid_y):
            pos_x = -width/2 + (x+0.5)*GRID_LEN
            pos_y = -height/2 + (y+0.5)*GRID_LEN
            pos_r = (pos_x**2 + pos_y **2 )**0.5            
            true_pos_r = pos_r*distance_rate
            #计算flux_map
            if pos_r <=compute_threshold:
                fit_flux[x,y] = attenuation_rate*fit_func(true_pos_r)/math.pi/compute_threshold/compute_threshold   # 4
            else:
                # 这一部分 空气的衰减因素已经 考虑在函数中
                fit_flux[x,y] = EnergyTransformDeriv(fit_func,func_distance,predict_diatance,pos_r)/2/math.pi/pos_r
    # 修正小于0 的部分以及峰值
    fit_flux[fit_flux<0] = 0
    return fit_flux

def plot3DFluxMap(flux_map,width = 10.05,height = 10.05):
    fig3d = plt.figure('3D plot')
    ax = Axes3D(fig3d)
    grid_x = round(width/GRID_LEN)
    grid_y = round(height/GRID_LEN)
    x = range(grid_x)
    y = range(grid_y)
    X,Y = np.meshgrid(x,y)
    Z = flux_map[X,Y]
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.jet) #hot
    plt.show()


# =============================================================================
# 这几个函数的主要目的是进行距离的变换
# =============================================================================
def AirAttenuation(func_distance):
    if(func_distance<1000):
        attenuation_rate =  0.99331-0.0001176*func_distance+1.97e-8*func_distance*func_distance;
    else:
        attenuation_rate =  math.exp(-0.0001106*func_distance)
    return attenuation_rate

PREDICT_MAX_R = 7.0
def EnergyTransform(func,func_distance,predict_diatance,predict_r):
    air_rate1 = AirAttenuation(func_distance)
    air_rate2 = AirAttenuation(predict_diatance)
    distance_rate = func_distance/predict_diatance;
    fun_r = predict_r*distance_rate;
    if fun_r>= PREDICT_MAX_R:
        fun_r = PREDICT_MAX_R
    energy = air_rate2/air_rate1*func(fun_r)
    return energy

def EnergyTransformDeriv(func,func_distance,predict_diatance,predict_r):
    air_rate1 = AirAttenuation(func_distance)
    air_rate2 = AirAttenuation(predict_diatance)
    distance_rate = func_distance/predict_diatance;
    fun_r = predict_r*distance_rate;
    if fun_r>= PREDICT_MAX_R:
        fun_r = PREDICT_MAX_R
    energy = air_rate2/air_rate1*func(fun_r,1)*distance_rate
    return energy

# =============================================================================
# 这个函数是使用其他距离的得到的CDF  来拟合PDF
# =============================================================================

def plotCDFDifference(fit_fun_train,train_distance,fit_fun_test,test_distance):
    # 确定是否要对于导数进行刻画
    plot_deriv = True
    
     #接下来要刻画的就是两个CDF的差距
    figConv = plt.figure('fig_conv')
    step_plot_r = [0.01*x for x in range(700)]
    plt.subplot(221)
    true_energy = fit_fun_test(step_plot_r)
    label_str = "{} m".format(test_distance)
    plt.plot(step_plot_r,true_energy,'b',lw =3,label= label_str)
    if plot_deriv:
        true_energy_daoshu = fit_fun_test(step_plot_r,1)
        plt.plot(step_plot_r,true_energy_daoshu,'c',lw =3,label= 'daoshu')
    plt.title('300m处拟合得到的CDF')
    
    plt.subplot(222)
    predict_energy = [EnergyTransform(fit_fun_train,train_distance,test_distance,x) for x in step_plot_r]
    label_str = "{}m 变换得到的结果".format(train_distance)
    plt.plot(step_plot_r,predict_energy,'r',lw =3,label= label_str)
    
    if plot_deriv:
        predict_energy_daoshu = [EnergyTransformDeriv(fit_fun_train,train_distance,test_distance,x) for x in step_plot_r]
        plt.plot(step_plot_r,predict_energy_daoshu,'c',lw =3,label= 'daoshu')
        
    plt.title('500m变换得到300mCDF')
    
    plt.subplot(223)
    label_str = "{} m".format(test_distance)
    plt.plot(step_plot_r,true_energy,'b',lw =3,label= label_str)
    label_str = "{}m 变换得到的结果".format(train_distance)
    plt.plot(step_plot_r,predict_energy,'r',lw =3,label= label_str)
    plt.title('Both')
    
    plt.subplot(224)
    plt.plot(step_plot_r,predict_energy - true_energy,'b',lw =3,label= "fit error")
    plt.title('两个CDF的差')
    figConv.show()    
    
def plotPDFDifference(true_pdf,fit_pdf,transform_pdf):
    #接下来要刻画的就是两个CDF的差距
    FluxMaxVal =  math.ceil(true_pdf.max()/50)*50
    fit_true_error = fit_pdf - true_pdf
    transform_true_error = transform_pdf - true_pdf
    transform_fit_error = transform_pdf - fit_pdf
     
    figConv = plt.figure('transform_PDF')
    plt.subplot(231)
    title_str = "{} m 处真实的PDF ".format(test_distance)
    plt.imshow(true_pdf, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=0, vmax=FluxMaxVal)
    plt.colorbar()
    plt.title(title_str)
    
    plt.subplot(232)
    title_str = "{} m 处拟合的PDF ".format(test_distance)
    plt.imshow(fit_pdf, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=0, vmax=FluxMaxVal)
    plt.colorbar()
    plt.title(title_str)
    
    plt.subplot(233)
    title_str = "{} m 变换得到的PDF ".format(train_distance)
    plt.imshow(transform_pdf, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=0, vmax=FluxMaxVal)
    plt.colorbar()
    plt.title(title_str)
    
    plt.subplot(234)
    title_str = "拟合与GT的差 "
    plt.imshow(fit_true_error, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=-20, vmax=20)
    plt.colorbar()
    plt.title(title_str)
    
    plt.subplot(235)
    title_str = "变换与 GT的差"
    plt.imshow(transform_true_error, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=-20, vmax=20)
    plt.colorbar()
    plt.title(title_str)
    
    plt.subplot(236)
    title_str = "变换与拟合的差"
    plt.imshow(transform_fit_error, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=-20, vmax=20)
    plt.colorbar()
    plt.title(title_str)
    figConv.show()

def plotConvDifference(true_conv_result,fit_conv_result,transform_conv_result):
    FluxMaxVal =  math.ceil(true_conv_result.max()/50)*50
    fit_true_error = fit_conv_result - true_conv_result
    transform_true_error = transform_conv_result - true_conv_result
    transform_fit_error = transform_conv_result - fit_conv_result
     
    figConv = plt.figure('conv_result')
    plt.subplot(231)
    title_str = "{} m 处真实的Flux map".format(test_distance)
    plt.imshow(true_conv_result, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=0, vmax=FluxMaxVal)
    plt.colorbar()
    plt.title(title_str)
    
    plt.subplot(232)
    title_str = "{} m 处拟合的Flux map".format(test_distance)
    plt.imshow(fit_conv_result, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=0, vmax=FluxMaxVal)
    plt.colorbar()
    plt.title(title_str)
    
    plt.subplot(233)
    title_str = "{} m 变换得到的Flux map".format(train_distance)
    plt.imshow(transform_conv_result, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=0, vmax=FluxMaxVal)
    plt.colorbar()
    plt.title(title_str)
    
    plt.subplot(234)
    title_str = "拟合与GT的差 "
    plt.imshow(fit_true_error, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=-20, vmax=20)
    plt.colorbar()
    plt.title(title_str)
    
    plt.subplot(235)
    title_str = "变换与 GT的差"
    plt.imshow(transform_true_error, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=-20, vmax=20)
    plt.colorbar()
    plt.title(title_str)
    
    plt.subplot(236)
    title_str = "变换与拟合的差"
    plt.imshow(transform_fit_error, interpolation='bilinear',origin='lower', \
               cmap =  cm.jet, vmin=-20, vmax=20)
    plt.colorbar()
    plt.title(title_str)
    figConv.show()


if __name__ == '__main__':
    process_angle = 0
    train_distance = 500
    test_distance = 300
    
    print('load the Data...')
    onepoint_path_train = DATA_PATH + "new/onepoint/{}/onepoint_angle_{}_distance_{}.txt".format(train_distance,process_angle,train_distance)
    onepoint_flux_train = np.genfromtxt(onepoint_path_train,delimiter=',')
    # 加载之前存储的函数
    print('count the flux data...')
    energy_static_train,energy_static_per_train,step_r_train = staticFlux(onepoint_flux_train,width=RECE_WIDTH,height = RECE_HEIGHT)
    print('accumulate the flux energy...')
    energy_accu_train = accumulateFlux(energy_static_train)
    fit_fun_train = getCDF(step_r_train,energy_accu_train,energy_static_per_train,1)
    
    print('load the Data...')
    onepoint_path_test = DATA_PATH + "new/onepoint/{}/onepoint_angle_{}_distance_{}.txt".format(test_distance,process_angle,test_distance)
    onepoint_flux_test = np.genfromtxt(onepoint_path_test,delimiter=',')
     
    # 加载之前存储的函数
    print('count the flux data...')
    energy_static_test,energy_static_per_test,step_r_test = staticFlux(onepoint_flux_test,width=RECE_WIDTH,height = RECE_HEIGHT)
    print('accumulate the flux energy...')
    energy_accu_test = accumulateFlux(energy_static_test)
    fit_fun_test = getCDF(step_r_test,energy_accu_test,energy_static_per_test,1)

#    # 使用RMSE来评价 误差函数
#    print("calculate the RMSE")
#    fit_rmse = rmse(fit_fun(step_r),energy_accu)
#    print('fit_RMSE: {} '.format(fit_rmse))
   
    # 展示距离变化 CDF    只是提供一个中间结果
    plotCDFDifference(fit_fun_train,train_distance,fit_fun_test,test_distance)
    
    # 计算 PDF 
    fit_pdf = getPDFFlux(fit_fun_test, width=RECE_WIDTH, height = RECE_HEIGHT)
    transform_pdf  = getPDFFluxTransform(fit_fun_train,train_distance,test_distance)
    # 计算拟合的误差
    total_energy = fit_pdf.sum()*GRID_LEN*GRID_LEN
    fit_total_energy = transform_pdf.sum()*GRID_LEN*GRID_LEN
    energy_rate = (fit_total_energy-total_energy)/total_energy*100
    print("RMSE: {}  Fit Energy: {}  fit error： {:.2f}%".format(total_energy,fit_total_energy,energy_rate))    
    # plot3DFluxMap(onepoint_flux)
    
    # 展示距离变化的 PDF   将两个或者三个PDF输出进行对比
    plotPDFDifference(onepoint_flux_test,fit_pdf,transform_pdf)
    
#    conv_path = DATA_PATH + "face/face_helio_31_distanct_{}.txt".format(test_distance)
#    true_conv_result = np.genfromtxt(conv_path,delimiter=',')
#    # 计算卷积积分
#    helio_range = np.zeros((200,200))
#    helio_range[90:110,70:130] = 1
#    fit_conv_result = signal.fftconvolve(helio_range, fit_pdf/400, mode='same')
#    transform_conv_result = signal.fftconvolve(helio_range, transform_pdf/400, mode='same')
#    ## 绘制卷积差异
#    plotConvDifference(true_conv_result,fit_conv_result,transform_conv_result)
    
    
    
    
#    print("fftconvolve cost {:.6f} s ".format(time_end-time_start))
#    FluxMaxVal =  math.ceil(conv_fft_result.max()/50)*50
#    figConv = plt.figure('fig_conv')
#    # plt.title("{} m结果对比".format(process_distance))
#    
#    plt.subplot(221)
#    plt.imshow(conv_fft_result, interpolation='bilinear',origin='lower', \
#               cmap =  cm.jet, vmin=0, vmax=FluxMaxVal)
#    plt.colorbar()
#    plt.title('卷积计算Flux Map')
#    
#    plt.subplot(222)
#    plt.imshow(conv_groundTruth, interpolation='bilinear',origin='lower', \
#               cmap =  cm.jet, vmin=0, vmax=FluxMaxVal)
#    plt.colorbar()
#    plt.title('Ray tracing 计算Flux Map')
#    
#    plt.subplot(223)
#    conv_err = conv_fft_result - conv_groundTruth
#    conv_err_max =  math.ceil(conv_err.max()/5)*5
#    conv_err_min =  math.floor(conv_err.min()/5)*5
#    plt.imshow(conv_fft_result - conv_groundTruth, interpolation='bilinear',origin='lower', \
#               cmap =  cm.jet, vmin=conv_err_min, vmax=conv_err_max)
#    plt.colorbar()
#    plt.title('Flux Map误差图')
    
    
    
    
    