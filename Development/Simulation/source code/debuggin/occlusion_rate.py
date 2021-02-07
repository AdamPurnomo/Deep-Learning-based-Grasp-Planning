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
        seg = seg.astype('float32')
        grasp_vector =  grasps[i,1] - grasps[i,0]
        grasp_center = ((grasps[i,0] + grasps[i,1]) / 2).astype(int)
        angle = np.arctan2(grasp_vector[1], grasp_vector[0]) * 180 / np.pi
        
        M = cv2.getRotationMatrix2D((grasp_center[0], grasp_center[1]), angle, 1.0)
        seg_im_val = cv2.warpAffine(seg, M, (seg.shape))
        left = int(grasp_center[0] - (width/2))
        right = int(grasp_center[0] + (width/2))    
        top = int(grasp_center[1] - (height/2))
        bottom = int(grasp_center[1] + (height/2))
        
        seg_bounding = seg_im_val[top:bottom, left:right]
        part_ID = ID
        part_indices = np.where(seg_bounding == part_ID)
        nonpart_indices = np.where(seg_bounding != part_ID)
        seg_bounding[part_indices[0], part_indices[1]] = 1
        seg_bounding[nonpart_indices[0], nonpart_indices[1]] = 0
        pix_num = np.sum(seg_bounding == 1)
        occlusion_rate_score = pix_num / constant
        occlusion_rate.append(occlusion_rate_score)
    
    return np.asarray(occlusion_rate)
    


""" This collision estimation function detects if there are pixel values which do not belong
to the part inside the grasp bounding box higher than those of pixel values that belong to the part. """




def collision_est(depth, seg, ID, grasps, image_size):
    collision_scores = []
    height = image_size[0]
    width = image_size[1]
    for i in range(len(grasps)):
        
        seg = seg.astype('float32')
        depth = depth.astype('float32')
        
        grasp_vector =  grasps[i,1] - grasps[i,0]
        grasp_length = np.linalg.norm(grasp_vector)
        grasp_center = ((grasps[i,0] + grasps[i,1]) / 2).astype(int)
        angle = np.arctan2(grasp_vector[1], grasp_vector[0]) * 180 / np.pi
        
        M = cv2.getRotationMatrix2D((grasp_center[0], grasp_center[1]), angle, 1.0)
        
        depth_im_val = cv2.warpAffine(depth, M, (depth.shape))
        seg_im_val = cv2.warpAffine(seg, M, (seg.shape))
        
        left = int(grasp_center[0] - (width/2))
        right = int(grasp_center[0] + (width/2))    
        top = grasp_center[1] - (height/2)
        bottom = grasp_center[1] + (height/2)

        #seg_full = seg_im_val[top:bottom, left:right]
        #depth_full = depth_im_val[top:bottom, left:right]

        part_ID = ID[i]
        seg_bounding = seg_im_val[grasp_center[1]-7:grasp_center[1]+7, left:right]
        depth_bounding = depth_im_val[grasp_center[1]-7:grasp_center[1]+7, left:right]
        
        part_indices = np.where(seg_bounding == part_ID) #in the form of [[row1, row2, .....], [column1, column2,.....]]
        nonpart_indices = np.where(seg_bounding != part_ID) #in the form of [[row1, row2, .....], [column1, column2,.....]]
        seg_bounding[part_indices[0], part_indices[1]] = 1
        seg_bounding[nonpart_indices[0], nonpart_indices[1]] = 0
        
        try:
            seg_mask = closing(seg_bounding, square(20))
            
            contact_point1 = int((width/2) - (grasp_length/2))
            contact_point2 = int((width/2) + (grasp_length/2))
            graspable_region1 = np.zeros((seg_mask.shape))
            graspable_region2 = np.zeros((seg_mask.shape))
            graspable_region1[:, contact_point1: contact_point1+5] = 1
            graspable_region2[:, contact_point2-5:contact_point2] = 1
            
            graspable_indices1 = np.where((seg_mask == 1)
                                        & (graspable_region1 ==1))
            graspable_indices2 = np.where((seg_mask == 1)
                                        & (graspable_region2 ==1))
            
            avgdepth_region1 = np.mean(depth_bounding[graspable_indices1[0], graspable_indices1[1]])
            avgdepth_region2 = np.mean(depth_bounding[graspable_indices2[0], graspable_indices2[1]])
            
            depth_region1 = np.copy(depth_bounding)
            depth_region1[:,int((width/2)):] = 1
            depth_region2 = np.copy(depth_bounding)
            depth_region2[:,0:int((width/2))] = 1
                                        
            if(np.isnan(avgdepth_region1)):
                cl_score1 = 10000
            else:
                cs_indices = np.where((seg_mask == 0) &
                                      (depth_region1 <= avgdepth_region1))
                cl_score1 = len(cs_indices[0])
        
            if(np.isnan(avgdepth_region2)):
                cl_score2 = 10000
            else:
                cs_indices = np.where((seg_mask == 0) &
                                      (depth_region2 <= avgdepth_region2))
                cl_score2 = len(cs_indices[0])
            
            if(cl_score1>30 or cl_score2>30):
                cl_score = 1
            else:
                cl_score = 0
            
            collision_scores.append(cl_score)
        except(IndexError):
            cl_score = 1
            collision_scores.append(cl_score)
            pass

    return np.asarray(collision_scores) 