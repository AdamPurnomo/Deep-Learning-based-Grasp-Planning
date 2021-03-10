# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:34:58 2020

@author: Adam Syammas Zaki P
"""

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D

friction_coef = 0.5
n_cone = 6
t_scal = 1/20
local_com = np.array([0,0,0])
origin_ws = np.array([0,0,0,0,0,0])

def min_distance_hull (p, cloud):
    """
Calculate the closest distance from a point to the convex hull facets.
'p' is the particular point that we want to calculate, should be an array of K-dimension,
cloud is the points that make up the hull (N points x K dimension).
"""

    #if (cloud.shape[1] > 3):
     #   model = PCA(n_components = cloud.shape[1]-1).fit(cloud)
      #  proj_cloud = model.transform(cloud)

    if not isinstance(cloud,ConvexHull):
        hull = ConvexHull(cloud, qhull_options = 'Qx')

    dist = []
    proj_p = [] #list of projected point on the hyperplanes
    
    for i in range(len(hull.equations)):
        n  = np.array(hull.equations[i,0:len(hull.equations[i])-1]) #return to the normal vector of the hyperplane
        s = np.dot(n, p) + hull.equations[i,len(hull.equations[i])-1]  #signed distance from test point to the hyperplane depending on the normal vector
        s_vec = s * n #vector from projected point on the hyperplane to the test point
        q = p - s_vec #projected point on the hyperplane
        q = np.hstack(q)
    
        dist.append(abs(s))
        proj_p.append(q)
        
    if(s > 0):  #if the distance is positive, the point is outside the convex hull
        q = 0
    else:
        q = min(dist) #/ max(dist)

    return q, dist.index(min(dist)), proj_p

def cone_wrenches(grasp, normal, com):
        cloud_w = [] #set wrenches
        vis_f = []
        tau = [] #set of torque
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
            
            h = 1 / (np.sqrt(1 + np.square(friction_coef)))
            r = h * friction_coef
            
            
            f = []
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
            vis_f.append(f)
        #g_force = [0,0,-9.8]
        #g_torque = [0,0,0]
        #g_wrench = np.hstack([g_force, g_torque])
        #cloud_w.append(g_wrench)
        cloud_w = np.reshape(cloud_w, (2*(n_cone+1), 6)) 
        #tau = np.reshape(tau, (2*n_cone, 3))
        #f = np.reshape(f, (2*n_cone, 3))
        #vis_f = np.reshape(vis_f, (2*n_cone, 3))
        return cloud_w, vis_f

for k in range(10):
    com = np.array([0,0,0])
    grasp = np.array([[-10, 0, k], [10, 0, -k]])
    normal = np.array([[1, 0, 0], [-1, 0, 0]])
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    r = [-10, 10]

    ax.plot([grasp[0][0],grasp[1][0]],[grasp[0][1],grasp[1][1]], [grasp[0][2],grasp[1][2]], 'k-')
    
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="b")
            
    cloud_w, force_cone = cone_wrenches(grasp, normal, local_com)
    q, index, proj = min_distance_hull(origin_ws, cloud_w)
    
    for i in range(len(force_cone)):
        for j in range(len(force_cone[i])):
            ax.quiver(grasp[i,0], grasp[i,1], grasp[i,2],force_cone[i][j][0],force_cone[i][j][1], force_cone[i][j][2],length=1)

    
            
    
    ax.set_title('Grasp Score = %f' %q)
    plt.show()
    print("Grasp Score = ", q)


