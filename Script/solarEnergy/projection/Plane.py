# -*- coding: utf-8 -*-
"""
Created on Mon May  7 20:05:14 2018

@author: nzx
"""

from vector import Vector
from Ray import Ray
import numpy as np

class Plane:
    """
        Plane gemometry
        
        right-hand coordinate 
        norm is z_axis
    """
    def __init__(self, pos, normal):
        self.pos = pos
        self.normal = normal   
        self.normal.normalize()
        self.is_setX = False
    
    def setX_axis(self,x_axis):
        self.is_setX = True
        self.x_axis = x_axis
        self.x_axis.normalize()
        self.y_axis = Vector.cross(self.normal,self.x_axis).normalize()

    def isIntersection(self, ray):
        """
        return isIntersect,intersect Point, distance
        """
        t = self.pos - ray.o
        a = Vector.dot(t, self.normal)
        b = Vector.dot(ray.d, self.normal)
        if abs(b) < 0.0001:
            return False, Vector(0.0, 0.0, 0.0), 0.0
        else:
            t = a / b
            if t > 0.0:
                return True, ray.o + ray.d * t, t
            else:
                return False, Vector(0.0, 0.0, 0.0), 0.0

    def getNormal(self):
        return self.normal
    
    def getPos(self):
        return self.pos
    
    def mapping(self,point):
        p_pos = point - self.pos
        x_pos = Vector.dot(p_pos,self.x_axis)
        y_pos = Vector.dot(p_pos,self.y_axis)
        return np.array([x_pos,y_pos])

if __name__ == "__main__":
    p_pos = Vector(0,0,0)
    p_norm = Vector(0,0,1)
    p_x_axis = Vector(1,0,0)
    p_norm.normalize()
    
    image_plane = Plane(p_pos,p_norm) 
    image_plane.setX_axis(p_x_axis)
    image_plane.y_axis.display()
    
    # test mapping
    ans = image_plane.mapping(Vector(1.5,2.5,4))
    print(ans)
    
    print("test the isIntersection")
    ray = Ray(Vector(1.4,2.4,3),Vector(0,0,-1))
    isInter,inter_p,dist = image_plane.isIntersection(ray)
    inter_p.display()
    

    