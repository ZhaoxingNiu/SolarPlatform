# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:59:58 2017

@author: nzx
"""

import pandas as pd
import numpy as np
#从excel中读取数据 保存成一个dataFrame
data = pd.read_excel("镜场位置数据.xlsx",sheet_name="pos",skiprows=0);
size_x  = 10.0
size_y  = 10.0
size_z = 10.0

inter_x = 10.0
inter_y = 10.0
inter_z = 10.0

helios_size_x = 2.66
helios_size_y = 0.10
helios_size_z = 1.88
                    
sizeLength = data.shape[0]
with open('helioField_small.txt','w') as f:
    for itemSize in range(0,sizeLength):
        f.writelines("# Grid%d attributes\n" % itemSize)
        f.writelines("Grid\t 0\n")
        f.writelines("pos\t\t%.6f\t%.6f\t%.6f\n" %(data.loc[itemSize].pos_x-5,data.loc[itemSize].pos_y-5,data.loc[itemSize].pos_z-5))
        f.writelines("size\t%.6f\t%.6f\t%.6f\n" %(size_x,size_y,size_z))
        f.writelines("inter\t%.6f\t%.6f\t%.6f\n" %(inter_x,inter_y,inter_z))
        f.writelines("n\t\t1\ntype\t0\nend\n\n")
        
        f.writelines("# Heliostats%d\n" % itemSize)
        f.writelines("gap\t\t0.200000\t0.200000\n")
        f.writelines("matrix\t1\t\t\t1\n")
        f.writelines("helio\t%.6f\t%.6f\t%.6f\n" %(data.loc[itemSize].pos_x,data.loc[itemSize].pos_y,data.loc[itemSize].pos_z))
        f.writelines("\t\t%.6f\t%.6f\t%.6f\n\n" %(helios_size_x,helios_size_y,helios_size_z))
