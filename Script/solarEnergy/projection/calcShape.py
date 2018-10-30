# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:38:17 2018

@author: nzx
"""
from vector import Vector,Vector2
from Heliostat import Heliostat
from Plane import Plane

import cv2
import math
import numpy as np

# 打印版本
print("opencv version: " + cv2.__version__)


def getImagePosition(res,plane):
    pos = np.zeros([4,2])
    if plane.is_setX == False:
        p1 = res[0][1]
        p4 = res[3][1]
        x_axis = p1 - p4
        plane.setX_axis(x_axis)
    for i in range(4):
        if(res[i][0]==False):
            print("the heliostat can not be projected in the plane")
        x_y = plane.mapping(res[i][1])
        pos[i][0] = x_y[0]
        pos[i][1] = x_y[1]
    return pos

def getImageCoor(pos,width = 10,height = 10,grid_size = 0.05):
    coor_list = np.zeros([4,2],np.int32)
    for i in range(4):
        x_index = int((pos[i][0] + width/2)/grid_size)
        y_index = int((pos[i][1] + height/2)/grid_size)
        coor_list[i][0] = x_index
        coor_list[i][1] = y_index
    # to crorrect the image plane
    """
        -1  -1
        -1   0
         0   0
         0  -1
    """
    coor_list[0][0] = coor_list[0][0] -1
    coor_list[0][1] = coor_list[0][1] -1
    coor_list[1][0] = coor_list[1][0] -1
    coor_list[3][1] = coor_list[3][1] -1
    return coor_list

def getArea(pos):
    """
    v1 = p1 - p0
    v2 = p4 - p0
    
    """
    print_pos = False
    if print_pos:
        print("pos:*********")
        print(pos)
    v1 = Vector2(pos[1][0] - pos[0][0],pos[1][1] - pos[0][1])
    v2 = Vector2(pos[3][0] - pos[0][0],pos[3][1] - pos[0][1])
    cos_theta = Vector2.dot(v1,v2)/v1.length()/v2.length()
    sin_theta = math.sqrt(1-cos_theta*cos_theta)
    area = v1.length()*v2.length()*sin_theta
    return area


def getCoor(heliostat,out_ray,plane):
    intersection = heliostat.castRay(out_ray,plane)
    pos = getImagePosition(intersection,plane)
    area = getArea(pos)
    coor = getImageCoor(pos)
    return coor,area


def calcShape(
        sun_ray = Vector(0.0,0.0,1.0),
        heliostat_pos = Vector(0.0,0.0,500.0),
        heliostat_size = Vector(3.0,0.1,1.0),
        receiver_center = Vector(0,0,0),
        receiver_x_axis = Vector(1,0,0),
        receiver_norm = Vector(0,0,1)):
  

    # get the reflect Ray
    sun_ray = sun_ray.normalize()
    out_ray = (receiver_center - heliostat_pos).normalize()
    norm_dir = (out_ray - sun_ray).normalize()
    distance = (receiver_center-heliostat_pos).length()
    shape_angle = Vector.angle(sun_ray*(-1),out_ray)
    

    heliostat = Heliostat(heliostat_pos,heliostat_size)
    heliostat.rotateTo(norm_dir)
    #heliostat.display()
    
    receiver_plane = Plane(receiver_center,receiver_norm)
    receiver_plane.setX_axis(receiver_x_axis)
    receiver_coor, receiver_area= getCoor(heliostat,out_ray,receiver_plane)
    

    image_plane = Plane(receiver_center,out_ray*(-1))
    image_coor,image_area = getCoor(heliostat,out_ray,image_plane)
    
    # 因为是单位向量，因此
    energy_attenuation = Vector.dot(receiver_plane.getNormal(),image_plane.getNormal())
    return image_coor,receiver_coor,distance,shape_angle,energy_attenuation,image_area,receiver_area


def calcBlock(
        sun_ray = Vector(0.0,0.0,1.0),
        heliostat_pos = Vector(0.0,0.0,500.0),
        block_heliostat_pos = Vector(0.0,0.0,498.0),
        heliostat_size = Vector(3.0,0.1,1.0),
        receiver_center = Vector(0,0,0),
        receiver_x_axis = Vector(1,0,0),
        receiver_norm = Vector(0,0,1)):
  
    
    # get the reflect Ray
    sun_ray = sun_ray.normalize()
    out_ray = (receiver_center - block_heliostat_pos).normalize()
    image_out_ray = (receiver_center - heliostat_pos).normalize()
    norm_dir = (out_ray - sun_ray).normalize()
    
    heliostat = Heliostat(heliostat_pos,heliostat_size)
    heliostat.rotateTo(norm_dir)
    #heliostat.display()

    image_plane = Plane(receiver_center,image_out_ray*(-1))
    image_coor,image_area = getCoor(heliostat,out_ray,image_plane)
    return image_coor,image_area



if __name__ =='__main__':
    print("calc shape...")

    receiver_center = Vector(0,0,0)
    receiver_x_axis = Vector(1,0,0)
    receiver_norm = Vector(0,0,1)

    heliostat_pos = Vector(0.0,0.0,500.0)
    heliostat_size = Vector(3.0,0.1,1.0)

    # get the reflect Ray
    sun_ray = Vector(0.0,0.0,1.0).normalize()
    out_ray = (receiver_center - heliostat_pos).normalize()
    norm_dir = (out_ray - sun_ray).normalize()

    heliostat = Heliostat(heliostat_pos,heliostat_size)
    heliostat.rotateTo(norm_dir)
    heliostat.display()
    
    print("define the plane and Image plane")
    receiver_plane = Plane(receiver_center,receiver_norm)
    receiver_plane.setX_axis(receiver_x_axis)
    receiver_coor = getCoor(heliostat,out_ray,receiver_plane)
    
#    receiver_intersection = heliostat.castRay(out_ray,receiver_plane)
#    receiver_pos = getImagePosition(receiver_intersection,receiver_plane)
#    receiver_coor = getImageCoor(receiver_pos)

    image_plane = Plane(receiver_center,out_ray*(-1))
    image_coor = getCoor(heliostat,out_ray,image_plane)
    
    # 因为是单位向量，因此
    energy_attenuation = Vector.dot(receiver_plane.getNormal(),image_plane.getNormal())

    print("get the Shape content")
#    np_image_plane = np.zeros((200,200),np.float)
#    cv2.fillPoly(np_image_plane, [image_coor], 1)
    
    
    