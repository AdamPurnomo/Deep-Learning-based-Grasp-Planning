# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:33:14 2019

@author: Adam Syammas Zaki P
"""
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from itertools import product, combinations
#from scipy.spatial import ConvexHull

from Wrench_Space_Analysis import min_distance_hull, viz_3D_hull

friction_coef = 0.5
n_cone = 6
t_scal = 1/10

def cone_wrenches(grasp, normal, com):
    
    cloud_w = [] #set wrenches
    #vis_f = []
    tau = [] #set of torque
    f = [] #set of force
    for i in range (len(grasp)):
        contact_p = grasp[i]
        n = normal[i]
        
        a = n[0]
        b = n[1]
        c = n[2]
        
        e1 = np.array([c-b, a-c, b-a])
        e1 = e1 / np.linalg.norm(e1)

        e2 = np.cross(n, e1)
        e2 = e2 / np.linalg.norm(e2)
       
        h = np.linalg.norm(n)
        r = h * friction_coef
        
        
        #f = []
        for j in range (n_cone):
            ei = h*n + r*np.cos(j*(2*np.pi/n_cone))*e1 + r*np.sin(j*(2*np.pi/n_cone))*e2 #Cone Direction
            alpha = 1 / (len(grasp)*n_cone) #constant for linear convex summation
            force = alpha*ei #Projection of normal force along the cone extrema
            torque = t_scal*np.cross(contact_p-com,force)
            wrenches = np.hstack([force, torque]) #generating wrenches 
            cloud_w.append(wrenches)
            if (j == n_cone-1):
                cloud_w.append(np.array([0,0,0,alpha*n[0],alpha*n[1],alpha*n[2]])) #additional wrenches for soft contact model for each contact point
            f.append(force)
            #force_debug.append(force)
            tau.append(torque)
        #vis_f.append(f)
    cloud_w = np.reshape(cloud_w, (2*(n_cone+1), 6)) 
    tau = np.reshape(tau, (2*n_cone, 3))
    f = np.reshape(f, (2*n_cone, 3))
    #force_debug = np.reshape(force_debug, (2*n_cone, 3))
    return cloud_w, f, tau


    
    
