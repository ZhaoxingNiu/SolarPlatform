# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:14:01 2018

@author: nzx

@descriiption: 定义了定日镜的坐标以及基本的旋转操作，需要与Ray保持一致
"""
from vector import Vector
import globalFunc
from Ray import Ray

__all__ = ['Heliostat']

class Heliostat:
    """
        the physial heliostat Face
            y+
            *
        p4*****p1
        ****O*******x+
        p3*****p2
        
        width is 8, height is 4 
        
    """
    def __init__(self,pos,size):
        self.pos = pos
        self.size = size
        self.p1 = Vector(-size.x/2,self.size.y/2,-size.z/2)
        self.p2 = Vector(self.p1.x,self.p1.y,self.p1.z+size.z)
        self.p3 = Vector(self.p2.x+size.x,self.p2.y,self.p2.z)
        self.p4 = Vector(self.p1.x+size.x,self.p1.y,self.p1.z)
        self.normal = Vector(0,1,0)
    
    
    def setCorner(self,p1,p2,p3,p4,normal):
        """
            使用直接赋值的方式进行计算坐标和法向
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.normal = normal
        
    def rotateTo(self,norm_dir):
        """
            坐标旋转这一部分需要参考Ray tracing部分的代码，需要保持一致
        """
        norm_dir.normalize() 
        self.normal = norm_dir
        self._posTransform()
        
    def _posTransform(self):
        """
            具体的坐标变换和旋转
        """
        self.p1 = globalFunc.local2World(self.p1,self.normal)
        self.p2 = globalFunc.local2World(self.p2,self.normal)
        self.p3 = globalFunc.local2World(self.p3,self.normal)
        self.p4 = globalFunc.local2World(self.p4,self.normal)
        
        self.p1 = globalFunc.transform(self.p1,self.pos)
        self.p2 = globalFunc.transform(self.p2,self.pos)
        self.p3 = globalFunc.transform(self.p3,self.pos)
        self.p4 = globalFunc.transform(self.p4,self.pos)
    
    def castRay(self,direction,plane):
        p1_ray = Ray(self.p1,direction)
        p2_ray = Ray(self.p2,direction)
        p3_ray = Ray(self.p3,direction)
        p4_ray = Ray(self.p4,direction)
        res = []
        res.append(plane.isIntersection(p1_ray))
        res.append(plane.isIntersection(p2_ray))
        res.append(plane.isIntersection(p3_ray))
        res.append(plane.isIntersection(p4_ray))
        return res
    
    def Area(self):
        return self.width*self.height
    
    def display(self):
        print("the corner position:")
        self.p1.display()
        self.p2.display()
        self.p3.display()
        self.p4.display()
        print("the normal:")
        self.normal.display()
    


if __name__ == "__main__":
    #define the heliostat
    print("simulation...")
    receiver_center = Vector(0,0,0)
    sun_ray = Vector(0,0.0,1.0)
    sun_ray.normalize()
    
    heliostat = Heliostat(Vector(0.0,0.0,500.0),Vector(3.0,0.1,1.0))
    heliostat.rotateTo(sun_ray*(-1))
    heliostat.display()