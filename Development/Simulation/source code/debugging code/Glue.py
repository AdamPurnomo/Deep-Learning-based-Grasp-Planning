# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:07:39 2019
@author: Adam Syammas Zaki P
"""

import numpy as np
import scipy as sp
import pybullet as pb
import pybullet_data
import matplotlib.pyplot as plt
import random as r
import time
import load_object as lo 
import wrench_space_analysis as wsa
import img_processor as ip
import occlusion_rate as ocr
import transformation as t
import collision_estimation as cst

"""
Note on parameters dictionary. This parameters describe the parameters belong to each specific part. 
The number below represents the index of the dictionary, and its respective description
0. min grasp distance in pixel
1. max grasp distance in pixel
2. min grasp distance in local coordinate of respective part
3. max grasp distance in local coordinate of respective part
4. Image heigth for each grasp candidate from respective part
5. Image width for each grasp candidate from respective part
6. Total number of pixel when the respective part is not occluded, viewed fromt the camera
7. The primitive geometry of the respective part
8. Directory of .obj format file of the respective part
9. Directory of .stl format file fo the respective part
10. Directory for saving the synthetic training data for the respective part
11. Directory for saving the synthetic training data for the respective part
"""
pi = np.pi/180
parameters = {'hv18': [8, 12, 0.02, 0.025, 60, 24, 500,
                       'cylindrical',
                       r'..\..\..\data\model\hv18_2.obj', 
                       r'..\..\..\data\model\hv18.stl',
                       r'..\..\..\Simulation\Image\Training Data\Classification\hv18\Positive\ ',
                       r'..\..\..\Simulation\Image\Training Data\Classification\hv18\Negative\ '], 
              'hv8':[38, 44, 0.09, 0.11, 60, 60, 1343,
                     'cicular',
                     r'..\..\..\data\model\hv8.obj', 
                     r'..\..\..\data\model\hv8.stl',
                     r'..\..\..\Simulation\Image\Training Data\Classification\hv8\Positive\ ',
                     r'..\..\..\Simulation\Image\Training Data\Classification\hv8\Negative\ ']}

        
def main(part_name):
     
    #parmeters initialization
    part_parameters = parameters[part_name]
    volumegrasp_range = [part_parameters[2], part_parameters[3]]    
    image_size = [part_parameters[4], part_parameters[5]]
    nonoccluded_pixel = part_parameters[6]
    geometry = part_parameters[7]
    part_dir = [part_parameters[8], part_parameters[9]]
    save_dir = [part_parameters[10], part_parameters[11]]
    
    #loading local grasps and scores data
    local_grasps = np.load(r'./local grasps/'+part_name+r'/local_grasps.npy')
    grasp_scores = np.load(r'./local grasps/'+part_name+r'/grasp_scores.npy')
    
    #initializing rendering parameters
    far = 6.1
    aspect = 1
    near = 0.1
    fov = 45.0
    img_size = [224, 224]
    renderingParameters = [far, near, aspect, fov, img_size]
    
    #opening simulation client
    physicsClient = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    #standard unit in simulation is set to be dm
    pb.setGravity(0,0,-98) 
    
    #loading object
    pb.loadURDF('plane.urdf')
    lo.load_container()
    partID = []
    num_parts = np.random.randint(5, 20)
   
    #simulation
    for i in range (num_parts):
        position = [0,0,i+1]
        orientation = [r.uniform(-pi*360,pi*360), r.uniform(-pi*360,pi*360), r.uniform(-pi*360,pi*360)]
        partID.append(lo.load_parts(position, orientation, part_dir))
        
    for i in range (1000):
        pb.stepSimulation(physicsClient)
        time.sleep(1./240.)

    #image rendering        
    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=[0, 0, 6],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0])
            
    projectionMatrix = pb.computeProjectionMatrixFOV(
                fov=fov,
                aspect=aspect,
                nearVal=near,
                farVal=far)
            
    width, height, rgbImg, depthbuff, segImg = pb.getCameraImage(
                width=img_size[1], 
                height=img_size[0],
                viewMatrix=viewMatrix,
                projectionMatrix=projectionMatrix)
                
    viewMatrix = np.reshape(viewMatrix, (4,4), 'F')
    projectionMatrix = np.reshape(projectionMatrix, (4,4), 'F')
    rendering_matrices = [viewMatrix, projectionMatrix]
            
    grasp_candidates = []
    u_grasp_candidates = []
    g_scores = []
    l_scores = []
    ID = []
    for i, ID in enumerate(partID):
        position, orientation = pb.getBasePositionAndOrientation(ID)
        rot_matrix = np.reshape(pb.getMatrixFromQuaternion(orientation), [3,3])
        T_matrix = np.zeros((4,4))
        T_matrix[3,3] = 1
        T_matrix[0:3,0:3] = rot_matrix
        T_matrix[0:3,3] = position

        #sampling grasps for each object
        #each object will have 10 grasp candidates
        indices = np.random.randint(0, len(local_grasps), size = 10)
        sampled_grasps = local_grasps[indices]
        sampled_graspscores = grasp_scores[indices]

        u_grasps = t.grasp_univ_transformation(T_matrix, sampled_grasps)
        p_grasps = t.to_pixel(u_grasps, rendering_matrices, img_size)
        
        twoDcom = t.pointpixel_transformation(np.array(position).reshape((1,3)), image_parameters)
        graspcenter = (twoDgrasps[:,0] + twoDgrasps[:,1]) / 2
        relative_dist = 2*np.linalg.norm(graspcenter - twoDcom, axis = 1)/image_size[0]
        l_scores.append(1-relative_dist)

        
        identity = np.zeros((10))
        identity[:] = partID[i]
        
        
        u_grasp_candidates.append(u_grasps)
        grasp_candidates.append(twoDgrasps)
        g_scores.append(r_graspscores)
        ID.append(identity)
    
    u_grasp_candidates = np.vstack(u_grasp_candidates)
    grasp_candidates = np.vstack(grasp_candidates)
    g_scores = np.hstack(g_scores)
    l_scores = np.hstack(l_scores)
    ID = np.hstack(ID)
   
    depth_loc = r'C:\Users\Furihata\Desktop\Adam\Grasp_Planning\Simulation\Image\depth.png'
    plt.imsave(depth_loc, depthImg, cmap='gray_r')
    pb.disconnect()
    
    c_scores = cst.collision_est(depthImg, segImg, ID, grasp_candidates ,image_size)
    o_scores = ocr.occlusion_rate(segImg, ID, grasp_candidates, image_size, nonoccluded_pixel)
    scores = np.array([g_scores, l_scores, o_scores, c_scores])
    ip.img_generator(grasp_candidates, depth_loc, scores, image_size, save_dir)
    
    np.save('depth.npy', depthImg)
    np.save('seg.npy', segImg)
    np.save('grasps.npy', grasp_candidates)
    np.save('ID.npy', ID)
    np.save('scores.npy', scores)

    #return grasp_candidates, scores
    

#iterate the program

for i in range(1):
    print("Iteration:", i)
    main('hv8')
    #grasp_candidates, scores, com = main('hv8')