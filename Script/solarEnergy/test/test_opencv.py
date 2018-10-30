# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:21:46 2018

@author: nzx
"""

import cv2
import numpy as np

# 打印版本
print(cv2.__version__)

# load Image
img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)
#img = img*1.0/500

#res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
rows,cols = img.shape
pts1 = np.float32([[30,40],[400,500],[0,440]])
pts2 = np.float32([[0,0],[460,690],[0,690 ]])
M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows),cv2.INTER_LINEAR)

#M = cv2.getPerspectiveTransform(pts1,pts2)
#res = cv2.warpPerspective(img,M,(200,200))

# display the image 
showImge = True
if showImge:
    cv2.imshow('image',dst)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# write a Image
cv2.imwrite('gray.png',img)


#access and mopdify the pixel value
pixel = img[100,100]
print(pixel)

# show image info
# shapoe = size + channel
print("size: "+ str(img.size))
print("shape: "+ str(img.shape))
print("size: "+ str(img.size))


#performance Measurement and improvement 

time1 = cv2.getTickCount()

time2 = cv2.getTickCount()
totaltime = (time2-time1)/cv2.getTickFrequency()
print("经过了："+ str(totaltime) + " s")