# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:15:16 2018

@author: nzx

@description: 定义了一个基本的光线的类
"""

class Ray:
    """
        Ray used to cast ray to tracer
    """
    def __init__(self, origin, direction):
        self.o = origin
        self.d = direction
