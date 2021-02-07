# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:40:55 2019

@author: Adam Syammas Zaki P
"""
# %%
import numpy as np
import trimesh as tr


# %%
def ray_tracing(local_grasps, stl):
    '''

    '''
    mesh = tr.load_mesh(stl, process = 'false')
    size = mesh.bounding_box.primitive.extents
    
    mesh.vertices -= mesh.center_mass 
    normal_surface = []
    nongrasps_indexes = []
    exact_grasps = []

    for i in range(len(local_grasps)):
        g0 = np.repeat(np.reshape(local_grasps[i,0], (1,3)), 21, axis=0) #generating 21 ray candidates for left grasp
        g1 = np.repeat(np.reshape(local_grasps[i,1], (1,3)), 21, axis=0) #generating 21 ray candidates for right grasp
        
        #Looking in which direction the rays are sampled over the object
        vec = local_grasps[i,0] - local_grasps[i,1] 
        dt_pr = np.array([abs(np.dot(vec, np.array([1,0,0]))),  abs(np.dot(vec, np.array([0,1,0]))), abs(np.dot(vec, np.array([0,0,1])))])
        index = np.argmin(dt_pr)
        dummy = np.zeros((21,))

        #obtaining exact location of local grasps and normal surface vector 
        for i in range(21):
            dummy[i] = (i-10)*size[index]/21
        g0[:,index] = dummy
        g1[:,index] = dummy
        
        ray0_origins = g0
        ray0_directions = g1-g0
        ray1_origins = g1
        ray1_directions = g0-g1
    
        #space_sample = np.reshape(np.hstack((g0,g1)), (-1,2,3))      
        
        p0, r0, t0 = mesh.ray.intersects_location(ray0_origins, ray0_directions, multiple_hits = False)
        p1, r1, t1 = mesh.ray.intersects_location(ray1_origins, ray1_directions, multiple_hits = False)
 

        if(len(p0) != 0 and len(p1)!= 0):
            r_int = np.intersect1d(r0, r1) #intersection of ray indexes between the left and right grasp
            if(len(r_int) == 0):
                p0 = np.reshape(np.array([0,0,0]),(1,3))
                p1 = np.reshape(np.array([0,0,0]),(1,3))
                exact_grasps.append([p0[0], p1[0]])
                normal_surface.append([p0[0] , p1[0]])
                nongrasps_indexes.append(i)
            else:
                r_argsort0 = np.argsort(r0)       #we're about to find the indices of element of intersection array in each ray's array
                r_argsort1 = np.argsort(r1)
                int_arg0 = r_argsort0[np.searchsorted(r0, r_int, sorter = r_argsort0)]
                int_arg1 = r_argsort1[np.searchsorted(r1, r_int, sorter = r_argsort1)]
                p0 = p0[int_arg0] #sorting the corrresponding point according to the intersection array
                p1 = p1[int_arg1]
                t0 = t0[int_arg0] #sorting the corresponding point according to the intersection array
                t1 = t1[int_arg1]
                
                dotpr = np.linalg.norm(p1-p0, axis = 1) #finding the longest valid grasps among the sampling candidates
                dummy_index = np.argmax(dotpr)
                exact_grasps.append([p0[dummy_index], p1[dummy_index]])
                normal_surface.append([mesh.face_normals[t0[dummy_index]] , mesh.face_normals[t1[dummy_index]]])
                
                
                
                #file_name = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\debug\Trimesh Scene\GraspsScene_' + str(i - len(nongrasps_indexes)) + '.png'
                #png = scene.save_image(resolution=[224, 224])
                #with open(file_name, 'wb') as f:
                #    f.write(png)
                #    f.close
                    
        else:
             p0 = np.reshape(np.array([0,0,0]),(1,3))
             p1 = np.reshape(np.array([0,0,0]),(1,3))
             exact_grasps.append([p0[0], p1[0]])
             normal_surface.append([p0[0] , p1[0]])
             nongrasps_indexes.append(i)           
    
        #if(np.linalg.norm(exact_grasps[i][0] - exact_grasps[i][1]) < 0.016):
            #nongrasps_indexes.append(i)
    
    nongrasps_indexes = np.asarray(nongrasps_indexes)
    if(len(nongrasps_indexes) == 0):
        exact_grasps = np.asarray(exact_grasps)
        normal_surface = np.asarray(normal_surface)
    else:
        exact_grasps = np.delete(np.asarray(exact_grasps), nongrasps_indexes, axis = 0)
        normal_surface = np.delete(np.asarray(normal_surface), nongrasps_indexes, axis = 0)
    
    return exact_grasps, normal_surface


    #Debugging
stl = r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\data\model\hv18.stl'
local_grasps = np.load('local_grasps.npy')
lx_grasps, normal, nn_indexes, size = ray_tracing(local_grasps, stl)

