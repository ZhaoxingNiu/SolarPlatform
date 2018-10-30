# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:46:30 2018

@author: nzx
"""
import math
from vector import Vector

EPLISION = 1e-6

def local2World(d_local,aligned_normal):
    """
    	// u : X
		// n is normal: Y
		// v : Z	
		// Sample(world coordinate)= Sample(local) * Transform matrix
		// Transform matrix:
		// |u.x		u.y		u.z|	
		// |n.x		n.y		n.z|	
		// |v.x		v.y		v.z|	
    
    """
    
    n = aligned_normal
    if(abs(n.x)<EPLISION and abs(n.z)<EPLISION):
        return d_local

# =============================================================================
#     if(abs(n.x)>abs(n.z)):
#         v = Vector.cross(n,Vector(0,1.0,0))
#         v.normalize()
#         u = Vector.cross(n,v)
#         u.normalize()
#     else:
# =============================================================================
    u = Vector.cross(Vector(0,1.0,0),n)
    u.normalize()
    v = Vector.cross(u,n)
    v.normalize()
    d_world = Vector(d_local.x*u.x + d_local.y*n.x + d_local.z*v.x,
                     d_local.x*u.y + d_local.y*n.y + d_local.z*v.y,
                     d_local.x*u.z + d_local.y*n.z + d_local.z*v.z)
    return d_world
        
def transform(d_in,transform_vec):
    return d_in + transform_vec


def rotateY(origin,old_dir,new_dir):
    """
       // Rotate matrix:
		// |cos		0		sin|	
		// |0		1		0  |	
		// |-sin	0		cos|	
    """
    dir =  1 if(Vector.cross(old_dir,new_dir).y >0) else -1
    cos_val = Vector.dot(old_dir,new_dir)
    sin_val = dir*math.sqrt(1-cos_val*cos_val)
    rotate_result = Vector(cos_val*origin.x + sin_val*origin.z,
                           origin.y,
                           -sin_val*origin.x + cos_val*origin.z)
    
    return rotate_result

if __name__ == "__main__":
    origin = Vector(1,0,1)
    aligned = Vector(0,0,1)
    rotated = local2World(origin,aligned)
    rotated.display()