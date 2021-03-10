# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:07:39 2019
@author: Adam Syammas Zaki P
"""
# %%
import sys 
sys.path.append(r'../../../utility')
import numpy as np
import scipy as sp
import pybullet as pb
import pybullet_data
import cv2
import random as r
import time
import load_object as lo 
import wrench_space_analysis as wsa
import transformation as t
import metrics as m 
import img_processor as ip
import mask
import json

"""
Note on parameters dictionary. This parameters describe the parameters belong to each specific target object. 
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
with open(r'..\JSON File\parameters.json') as f:
    parameters = json.load(f)

#%%
def main(part_name, visualize=False, scale=1):
    '''
    This function will start the simulation environment and generate training data
    of the target object and the 2D projection of grasp approaching vector of the positive class

    #input
    part_name       : The name of the target object (str)
    visualize       : Boolean whether to visualize the simulation or not
    scale           : Relative sclace of the 3D CAD data and the simulation environment
    '''
     
    #parmeters initialization
    part_parameters = parameters[part_name]
    volumegrasp_range = [part_parameters[2], part_parameters[3]]    
    data_size = [part_parameters[4], part_parameters[5]]
    nonoccluded_pixel = part_parameters[6]
    part_dir = [part_parameters[7], part_parameters[8]]
    save_dir = [part_parameters[9], part_parameters[10], part_parameters[11]]

    
    #loading local grasps and scores data
    local_grasps = np.load(r'../Npy File/local grasps/'+part_name+r'/local_grasps.npy')[0]
    grasp_scores = np.load(r'../Npy FIle/local grasps/'+part_name+r'/scores.npy')
    grasp_scores = 0.5*(grasp_scores[0]+grasp_scores[1])
    
    #initializing rendering parameters
    far = 8.7
    aspect = 1.25
    near = 7.3
    fov = 37
    img_size = [1024, 1280]
    renderingParameters = [far, near, aspect, fov, img_size]
    
    #opening simulation client
    if(visualize==True):
        physicsClient = pb.connect(pb.GUI)
    else:
        physicsClient = pb.connect(pb.DIRECT)
    
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    #standard unit in simulation is set to be dm
    pb.setGravity(0,0,-98) 
    
    #loading object
    pb.loadURDF('plane.urdf')
    lo.load_container()
    partID = []
    num_parts = np.random.randint(5, 25)
   
    #simulation
    for i in range (25):
        position = [0,0,i+2]
        orientation = [r.uniform(-pi*360,pi*360), r.uniform(-pi*360,pi*360), r.uniform(-pi*360,pi*360)]
        partID.append(lo.load_parts(position, orientation, part_dir,scale))
    
    for i in range (1000):
        pb.stepSimulation(physicsClient)
        time.sleep(1./240.)

    #image rendering        
    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=[0, 0, 9.1],
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


    p_grasps = [] #projected grasps in an image
    u_grasps = [] #transformed grasps in universal coordinate
    fc_scores = [] #ferrary canny grasp quality metrics
    graspID = []  # Associated object id with each grasp candidates
    u_zvec = [] #transformed grasp approaching p
    p_zvec = [] #projected z_vector
    

    for i, ID in enumerate(partID):
        position, orientation = pb.getBasePositionAndOrientation(ID)
        rot_matrix = np.reshape(pb.getMatrixFromQuaternion(orientation), [3,3])
        T_matrix = np.zeros((4,4))
        T_matrix[3,3] = 1
        T_matrix[0:3,0:3] = rot_matrix
        T_matrix[0:3,3] = position

        #sampling grasps for each object
        #each object will have 25 grasp candidates
        num_sample = 10
        indices = np.random.randint(0, len(local_grasps), size = num_sample)
        sampled_grasps = local_grasps[indices]
        sampled_graspscores = grasp_scores[indices]

        #generate mask for orientation estimation
        z_vec = mask.z_vec_transformation(rot_matrix, 
                                                    t.grasp_univ_transformation(T_matrix, sampled_grasps))
        z_vec = mask.to_pixel(z_vec, rendering_matrices, img_size)

        #grasps projection to an image
        u_grasps.append(t.grasp_univ_transformation(T_matrix, sampled_grasps))

        #remove clipped grasp candidates
        projected, clipped = t.to_pixel(u_grasps[i], rendering_matrices, img_size)
        sampled_graspscores = np.delete(sampled_graspscores, clipped, axis = 0)
        z_vec = np.delete(z_vec, clipped, axis = 0)

        p_grasps.append(projected)
        fc_scores.append(sampled_graspscores)
        graspID.append(np.broadcast_to(ID, (len(projected),)))
        p_zvec.append(z_vec)
        
    #reshaping tensor
    u_grasps = np.vstack(u_grasps)
    p_grasps = np.vstack(p_grasps)
    fc_scores = np.hstack(fc_scores)
    graspID = np.hstack(graspID)
    p_zvec = np.vstack(p_zvec)

    #processing depth image
    depthmap = t.to_depthMap(depthbuff, renderingParameters)
    depthImg = cv2.normalize(depthmap, 
                            dst=None, 
                            alpha=0, 
                            beta=255, 
                            norm_type=cv2.NORM_MINMAX, 
                            dtype=cv2.CV_8UC1)
   
    
    #collision estimation
    c_scores, debugging_cs = m.collision_est(depthmap, 
                                segImg, 
                                graspID, 
                                p_grasps,
                                data_size)

    #occlusion rate estimation
    o_scores = m.occlusion_rate(segImg, 
                                graspID, 
                                p_grasps, 
                                data_size, 
                                nonoccluded_pixel)

    scores = np.array([fc_scores, o_scores, c_scores]).T


    grasp_img, r_matrix = ip.data_generator(p_grasps,
                     depthImg,
                     data_size)

    #creating mask
    viz, vect, v_mat, s_mat, invalid_indices = mask.mask_visualization(p_zvec,
                                   p_grasps,
                                   data_size)

    #pruning images
    scores = np.delete(scores, invalid_indices, axis = 0)
    grasp_img = np.delete(grasp_img, invalid_indices, axis = 0)
    debugging_cs = np.delete(debugging_cs, invalid_indices, axis = 0)
    p_grasps = np.delete(p_grasps, invalid_indices, axis = 0)
    u_grasps = np.delete(u_grasps, invalid_indices, axis = 0)
    p_zvec = np.delete(p_zvec, invalid_indices, axis = 0)

    #visualizing grasps
    if(visualize==True):
        t.draw_realgrasps(u_grasps, fc_scores)

    #label grasp candidates
    labels = np.zeros(len(scores))
    p_index = np.where((scores[:,0]>0.7) &
                        (scores[:,1]>0.9) &
                        (scores[:,2]==0))
    labels[p_index] = 1

    full_grasp_img = ip.draw_grasp_representation(p_grasps, p_zvec, labels, depthImg, data_size)
    

    #discriminate and saving mask
    basedir = save_dir[0:2]
    nameid = ip.save_data(basedir,
                 labels,
                 grasp_img)
    
    basedir = save_dir[2]
    mask.save_mask(basedir, 
                   vect, 
                   v_mat, 
                   s_mat, 
                   scores, 
                   nameid)
    pb.disconnect()

if __name__ == "__main__":
    num = 1800
    for i in range(num):
        print("Iteration: ", i)
        main('hv6')



