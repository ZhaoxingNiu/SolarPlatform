# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:02:29 2018

@author: nzx

@description: vector operator
"""

import math


__all__ = ["Vector2", "Vector"]

class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    @staticmethod
    def dot(v0, v1):
        return v0.x * v1.x + v0.y * v1.y

    def __sub__(self, v):
        return Vector2(self.x - v.x, self.y - v.y)
    
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def display(self):
        print( "x: {:.2f} y: {:.2f}".format(self.x,self.y))


class Vector:
    def __init__(self,x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def dot(v0, v1):
        return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z
    
    @staticmethod
    def angle(v0, v1,radian =False):
        cos_val = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z
        if radian:
            return math.acos(cos_val)
        return math.acos(cos_val)*180/math.pi

    @staticmethod
    def cross(v0, v1):
        return Vector(v0.y * v1.z - v0.z * v1.y,
                      v0.z * v1.x - v0.x * v1.z,
                      v0.x * v1.y - v0.y * v1.x)

    @staticmethod
    def reflect(n, l):
        """
        :param n: 法向
        :param l: 点指向入射方向
        :return: 点指向出射方向
        """
        ln = Vector.dot(n, l)
        ln = 2 * ln
        n = n * ln
        return (n - l).normalize()

    @staticmethod
    def outRay(n, l):
        """
        :param n: 法向
        :param l: 入射方向指向点
        :return: 点指向出射方向
        """
        ln = Vector.dot(n, l)
        ln = 2 * ln
        n = n * ln
        return (n + l).normalize()
    
    def __add__(self, v):
        return Vector(self.x + v.x, self.y + v.y, self.z + v.z)

    def __sub__(self, v):
        return Vector(self.x - v.x, self.y - v.y, self.z - v.z)

    def __mul__(self, v):
        if isinstance(v,int) or isinstance(v,float):
            return Vector(self.x * v, self.y * v, self.z * v)
        elif isinstance(v,Vector):
            return Vector(self.x * v.x, self.y * v.y, self.z * v.z)
        else:
            raise Exception("Invalid type!")
            
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def lengthSqure(self):
        return self.x * self.x + self.y * self.y + self.z * self.z
        
    def normalize(self):
        l = self.length()
        if l == 0.0:
            l = 1.0
        self.x /= l
        self.y /= l
        self.z /= l
        return self
        
    def display(self):
        print( "x: {:.2f} y: {:.2f} z:{:.2f}".format(self.x,self.y,self.z))
        
if __name__ == "__main__":
    d = Vector(0, 1, 1)
    d.normalize()
    a = Vector(1.0, 0.0, 0.0)
    cos = Vector.dot(a,d)
    print(cos)
    angle = Vector.angle(a,d)
    print(cos)
    
    # test the right-hand coordinate
    x = Vector(1,0,0)
    y = Vector(0,1,0)
    z = Vector.cross(x,y)
    z.display()

    # test the reflect
    n = Vector(0,1,0)
    l = Vector(0,1,1).normalize()
    r = Vector.reflect(n,l)
    r.display()