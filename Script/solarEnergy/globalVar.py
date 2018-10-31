# -*- coding: utf-8 -*-
"""
Created on Wed May  9 21:22:20 2018

@author: nzx
"""

# 指定数据的位置以及输出的位置
DATA_PATH = "E:/program/solarEnergyGit/SolarPlatform/SimulResult/data/"
# 统计能量分布时 R 的bin的大小
STEP_R = 0.05
# GRID_LEN  对应的是接收面的格子边长
GRID_LEN = 0.05

# 不适用Threshold阈值的结果
DISTANCE_THRESHOLD = 0.1

# 接收面的长  接收面的宽
RECE_WIDTH = 10.05
RECE_HEIGHT = 10.05

# 接收面的半径
RECE_MAX_R = 7.0 

# 定日镜面积的大小
HELIOSTAT_AREA = 3.0

# config
IS_PLOT_CDF = False

# 是否展示evaluate的图像
IS_PLOT_EVALUATE = False


def re_init_para(args):
    STEP_R = args.step_r
    GRID_LEN = args.grid_len 
    DISTANCE_THRESHOLD = args.distance_threshold 
    RECE_WIDTH = args.rece_width 
    RECE_HEIGHT = args.rece_height 
    RECE_MAX_R = args.rece_max_r