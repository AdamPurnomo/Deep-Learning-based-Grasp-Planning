# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:46:38 2020

@author: Adam Syammas Zaki P
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import binary_closing, closing, square

def force_closure(grasps, normals, friction_coeff):
    '''
    Calculate the dot product of grasp vector relative to normal surface
    #input
    grasps      : set of grasp candidates in 3D coordinate space
                  Shape (n, 2, 3)
    normals     : set of normal surface vectors associated with each grasp candidates
                  Shape (n, 2, 3)
    friction_coeff  : friction coefficient to determine how large the friction cone is

    #output
    fc_score    : force closure score
    '''
    c1 = grasps[:,0,:]
    c2 = grasps[:,1,:]

    n1 = normals[:,0,:]
    n2 = normals[:,1,:]

    v =  c1 - c2
    v_norm = np.linalg.norm(v, axis=1)
    v = v / np.tile(v_norm[:, np.newaxis], [1, 3])

    ip1 = np.sum(n1 * v, axis=1)
    ip2 = np.sum(n2 * -v, axis=1)

    fc_score = (ip1 + ip2)/2
    return fc_score 


def occlusion_rate(segImg, graspID, grasps, image_size, constant):
    '''
    Calculate the percentage of object part that is still visible from the camera.
    The constant parameter is the total number of pixel in the image
    if the part is not occluded at all. This number is determined by performing a simulation
    only on a single part. The occulsion rate is calculated by dividing the number of availabe pixel
    that belongs to a specific part in the bin by the constant parameter.
    #input
    segImg      : segmentation image from simulation engine
                 Shape (m,n)
    graspID     : set of associated object ID with grasp candidates
                 Shape (n,)
    grasps      : set of grasp candidates in pixel space
                  Shape (n, 2, 2)
    image_size  : size of generated image as synthetic data
                  Shape (2,)
    constant    : total number of visible pixel if the object is not occluded

    #output
    occlusion_rate  : occlusion rate score for each grasp candidates
                      Shape (n, )

    '''
    height = image_size[0]
    width = image_size[1]
    occlusion_rate = []
    for i, ID in enumerate(graspID):        
        #processing the grasp bounding box
        part_ID = ID
        pix_num = np.sum(segImg == part_ID)
        occlusion_rate_score = pix_num / constant
        occlusion_rate.append(occlusion_rate_score)
    
    return np.asarray(occlusion_rate)
    

def collision_est(depthbuff, segImg, graspID, grasps, image_size):
    '''
    This collision estimation function detects if there are pixel values which do not belong
    to the part inside the grasp bounding box higher than those of pixel values that belong to the part.

    #input
    depthbuff        : depth buffer
                      Shape (m,n)
    segImg          : segmentation image from simulation engine
                      Shape (m,n)
    graspID         : set of associated object ID with grasp candidates
                     Shape (n,) 
    grasps          : set of grasp candidates in pixel space
                    Shape (n, 2, 2)
    image_size  : size of generated image as synthetic data
                    Shape (2,)
    
    #output
    collision_scores    : collision estimation score for each grasp candidates
                         Shape (n,)
    debugging_images    : segmentation images to see how many pixel indicates collision
                         Shape (220,n)

    '''
    collision_scores = []
    height = image_size[0]
    width = image_size[1]
    debugging_images = []
    for i, ID in enumerate(graspID):
        
        mask = np.zeros((40, width))
        segImg = segImg.astype('float32')
        depthImg = depthbuff
        
        grasp_vector =  grasps[i,1] - grasps[i,0]
        grasp_length = np.linalg.norm(grasp_vector)
        grasp_center = ((grasps[i,0] + grasps[i,1]) / 2).astype(int)
        angle = np.arctan2(grasp_vector[1], grasp_vector[0]) * 180 / np.pi
        
        #rotation and cropping
        M = cv2.getRotationMatrix2D((grasp_center[0], grasp_center[1]), angle, 1.0)
        depth_im_val = cv2.warpAffine(depthImg, M, (depthImg.shape))
        seg_im_val = cv2.warpAffine(segImg, M, (segImg.shape))
        left = int(grasp_center[0] - (width/2))
        right = int(grasp_center[0] + (width/2))    
        top = grasp_center[1] - (height/2)
        bottom = grasp_center[1] + (height/2)

        part_ID = ID
        seg_bounding = seg_im_val[grasp_center[1]-20:grasp_center[1]+20, left:right]
        depth_bounding = depth_im_val[grasp_center[1]-20:grasp_center[1]+20, left:right]
        
        part_indices = np.where(seg_bounding == part_ID) 
        nonpart_indices = np.where(seg_bounding != part_ID) 
        seg_bounding[part_indices[0], part_indices[1]] = 1
        seg_bounding[nonpart_indices[0], nonpart_indices[1]] = 0
        

        try:
            seg_mask = closing(seg_bounding, square(20))
            
            contact_point1 = int((width/2) - (grasp_length/2))
            contact_point2 = int((width/2) + (grasp_length/2))
            graspable_region1 = np.zeros((seg_mask.shape))
            graspable_region2 = np.zeros((seg_mask.shape))
            graspable_region1[:, contact_point1: contact_point1+10] = 1
            graspable_region2[:, contact_point2-10:contact_point2] = 1
            
            graspable_indices1 = np.where((seg_mask == 1)
                                        & (graspable_region1 ==1))
            graspable_indices2 = np.where((seg_mask == 1)
                                        & (graspable_region2 ==1))
            
            avgdepth_region1 = np.mean(depth_bounding[graspable_indices1[0], graspable_indices1[1]])
            avgdepth_region2 = np.mean(depth_bounding[graspable_indices2[0], graspable_indices2[1]])
            
            depth_region1 = np.copy(depth_bounding)
            depth_region1[:,int((width/2)):] = 100
            depth_region2 = np.copy(depth_bounding)
            depth_region2[:,0:int((width/2))] = 100
                                        
            if(np.isnan(avgdepth_region1)):
                cl_score1 = 10000
            else:
                cs_indices = np.where((seg_mask == 0) &
                                      (depth_region1 <= avgdepth_region1+0.05))
                mask[cs_indices] = 1
                cl_score1 = len(cs_indices[0])
        
            if(np.isnan(avgdepth_region2)):
                cl_score2 = 10000
            else:
                cs_indices = np.where((seg_mask == 0) &
                                      (depth_region2 <= avgdepth_region2+0.05))
                mask[cs_indices] = 1
                cl_score2 = len(cs_indices[0])

            if(cl_score1>100 or cl_score2>100):
                cl_score = 1
            else:
                cl_score = 0
            
            collision_scores.append(cl_score)
        except(IndexError, ValueError):
            cl_score = 1
            collision_scores.append(cl_score)
            pass
        debugging_images.append(mask)

    return np.asarray(collision_scores), debugging_images 