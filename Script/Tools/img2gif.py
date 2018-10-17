# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 12:55:15 2017

@author: nzx
"""

from PIL import Image
import os

filePath = 'C:/Users/Administrator/Desktop/data/face_conv/'
filenames=sorted((fn for fn in os.listdir(filePath) if fn.endswith('.png')))

im = Image.open(filePath+filenames[0])
images = []
for filename in filenames:
    print(filePath+filename)
    images.append(Image.open(filePath+filename))
#imageio.mimsave('C:/Users/Administrator/Desktop/data/gif.gif', images_list,duration=1)
im.save(filePath+'gif.gif', save_all=True, append_images=images,loop=1,duration=500)