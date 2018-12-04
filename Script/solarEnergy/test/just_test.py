# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:07:21 2018

@author: nzx
"""

blue_line = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=15, label='Blue stars')
plt.legend(handles=[blue_line])

plt.show()