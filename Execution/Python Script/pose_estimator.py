# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:28:21 2020

@author: Adam Syammas Zaki Purnomo
"""
import numpy as np 

def locate_vector_matrix(masks):
    '''
    Locate the pixel location in vector matrix 
    where it contains the information of grasp approaching vector

    #input
    masks : set of masks from the prediction of the neural network 
            Shape (n, height, width)

    #output
    rel_vect : grasp approaching vector projection in 2d pixel coordinate relative 
               to the center of the grasp representation image
               Shape (n,2)
    '''
    height, width = masks.shape[1:3]
    img_center = np.array([width/2, height/2], dtype = 'int32')
    rel_vect = []
    for mask in masks:
        vect = np.argwhere(mask==1)
        vect = np.flip(vect, axis=1)
        vect = vect - img_center
        norm = np.linalg.norm(vect,axis=1)
        max_index = np.argmax(norm)
        vect = vect[max_index]
        rel_vect.append(vect)
    return rel_vect

def locate_sparse_matrix(masks):
    '''
    Locate the pixel location in sparse matrix
    where it contains the information of grasp approaching vector

    #input
    masks : set of masks from the prediction of the neural network 
            Shape (n, height, width)

    #output
    rel_vect : grasp approaching vector projection in 2d pixel coordinate relative 
               to the center of the grasp representation image
               Shape (n,2)
    '''
    height, width = masks.shape[1:3]
    img_center = np.array([width/2, height/2], dtype = 'int32')
    max_indices = masks.reshape(masks.shape[0], -1).argmax(axis=1)
    vect = np.column_stack(np.unravel_index(max_indices, masks.shape[1:3]))
    vect = np.flip(vect, axis=1)
    vect = vect - img_center
    return vect

def rotate_back(rel_vect, M, p_grasps):
    '''
    transform back the projection of grasp approaching vector from grasp representation image to
    the scene image
    
    #input
    rel_vect    : grasp approaching vector projection in 2d pixel coordinate relative 
                  to the center of the grasp representation image
                  Shape (n,2)
    M           : 2D rotation vector
                 Shape (n,2,3)
    p_grasps    : positive grasp candidates in pixel coordinate
                 Shape (n,2,2)
    #ouput
    ga_vect     : set of grasp approaching vector in 2d pixel coordinate of the scene image
    '''
    size = len(p_grasps)
    ga_vect = []
    for i in range(size):
        center = ((p_grasps[i,0] + p_grasps[i, 1]) / 2).astype(int)
        vect = center + rel_vect[i]
        vect = np.array([vect[0], vect[1], 1])

        R = np.eye(3)
        R[0:2, 0:3] = M[i]
        Rinv = np.linalg.inv(R)

        vect = Rinv.dot(vect)
        ga_vect.append(vect)
    ga_vect = np.array(ga_vect, dtype = 'int32')
    ga_vect = ga_vect[:, 0:2]
    return ga_vect

def apv_to_cartesian(pointmap, ga_vect, p_grasps, scale=0.25):
    '''
    Transform grasp approaching vector from pixel coordinate to
    cartesian cooridanate
    #input
    pointmap : pointmap of the scene
               shape (3, m, n)
    ga_vect   : projection of grasp approaching vector in pixel coordinate
             shape (n, 2)    
    p_grasps   : positive grasp candidates in pixel coordinate
              shape (n, 2, ,2)
    scale    : the norm of the grasp approaching vector

    #output
    cart_apvect  : approaching pose vector in 3d cartesian coordinate  
    '''
    xmap = pointmap[0]
    ymap = pointmap[1]
    zmap = pointmap[2]
    center = ((p_grasps[:, 0] + p_grasps[:,1]) / 2).astype(int)
    size = len(ga_vect)
    cart_apvect = []
    for i in range(size):
        column = ga_vect[i,0].astype(int)
        row = ga_vect[i,1].astype(int)

        x = xmap[row, column]
        y = ymap[row, column]
        
        c_column = center[i,0]
        c_row = center[i,1]
        c_x = xmap[c_row, c_column]
        c_y = ymap[c_row, c_column]
        c_z = zmap[c_row, c_column]

        delta_x = -(x - c_x)
        delta_y = -(y - c_y)

        dummy = scale**2 - (delta_x)**2 - (delta_y)**2
        delta_z = np.sqrt(dummy)
        z = c_z + delta_z
        
        a_vec = np.array([delta_x, delta_y, delta_z])
        cart_apvect.append(a_vec)
        
    cart_apvect = np.array(cart_apvect)
    return cart_apvect


def Z_R(psi):
    '''
    Rotation matrix around Z-axis 
    
    #input
    psi     : rotation angle
    #output
    Z       : rotation matrix around z axis
    '''
    Z = np.eye(3)
    Z[0,0] = np.cos(psi)
    Z[0,1] = -np.sin(psi)
    Z[1,0] = np.sin(psi)
    Z[1,1] = np.cos(psi)
    return Z

def Y_R(theta):
    '''
    Rotation matrix around Y-axis 
    
    #input
    theta     : rotation angle
    #output
    Y       : rotation matrix around y axis
    '''
    Y = np.eye(3)
    Y[0,0] = np.cos(theta)
    Y[0,2] = np.sin(theta)
    Y[2,0] = -np.sin(theta)
    Y[2,2] = np.cos(theta)
    return Y

def X_R(phi):
    '''
    Rotation matrix around X-axis 
    
    #input
    phi     : rotation angle
    #output
    X       : rotation matrix around x axis
    '''
    X = np.eye(3)
    X[1,1] = np.cos(phi)
    X[1,2] = -np.sin(phi)
    X[2,1] = np.sin(phi)
    X[2,2] = np.cos(phi)
    return X

def extract_pose_v1(p_grasps, cart_apvect, scale):
    '''
    Extract euler angle from approaching pose vector

    #input
    p_grasps        : positive grasp candidates in pixel coordinate
    cart_apvect     : grasp approaching vector in cartesian coordinate  
    scale           : the norm of grasp approaching vector

    #output
    rot_matrix      : rotation matrix on how to approach the object
                      Shape (n,3,3)
    euler           : euler rotation on how to approach the object
                     [yaw, pitch, roll]
                     Shape (n,3)        
    '''
    size = len(p_grasps)
    rot_matrix = []
    euler_angle = []

    for i in range(size):
        vect1 = p_grasps[i,0] - p_grasps[i,1]
        vect2 = p_grasps[i,1] - p_grasps[i,0]
        psi1 = np.arctan2(-vect1[1], vect1[0])
        psi2 = np.arctan2(-vect2[1], vect2[0])
        psi = np.array([psi1, psi2])
        indices = np.argmin(np.abs(psi))
        psi = psi[indices]

        Z = Z_R(psi)
        Zinv = np.linalg.inv(Z)
        v = Zinv.dot(cart_apvect[i]/scale)

        phi = np.arcsin(-v[1])
        theta = np.arcsin(v[0]/np.cos(phi))
        Y = Y_R(theta)
        X = X_R(phi)

        R = Z.dot(Y.dot(X))

        rot_matrix.append(R)
        euler_angle.append(np.array([phi, theta, psi]))
    return rot_matrix, np.array(euler_angle)

def extract_pose_v2(p_grasps, cart_apvect, scale):
    '''
    Extract euler angle from approaching pose vector

    #input
    p_grasps        : positive grasp candidates in pixel coordinate
    cart_apvect     : grasp approaching vector in cartesian coordinate  
    scale           : the norm of grasp approaching vector

    #output
    rot_matrix      : rotation matrix on how to approach the object
                      Shape (n,3,3)
    euler           : euler rotation on how to approach the object
                     [yaw, pitch, roll]
                     Shape (n,3)        
    '''
    size = len(p_grasps)
    rot_matrix = []
    euler_angle = []

    for i in range(size):
        vect1 = p_grasps[i,0] - p_grasps[i,1]
        vect2 = p_grasps[i,1] - p_grasps[i,0]
        psi1 = np.arctan2(-vect1[1], vect1[0])
        psi2 = np.arctan2(-vect2[1], vect2[0])
        psi = np.array([psi1, psi2])
        indices = np.argmin(np.abs(psi))
        psi = psi[indices]

        Z = Z_R(psi)
        v = cart_apvect[i]/scale

        phi = np.arctan2(-v[1],v[2])
        theta = np.arcsin(v[0])
        Y = Y_R(theta)
        X = X_R(phi)

        R = X.dot(Y.dot(Z))

        rot_matrix.append(R)
        euler_angle.append(np.array([phi, theta, psi]))
    return rot_matrix, np.array(euler_angle)


def pre_grasp_pos_v1(grasp_pos,euler,dist):
    '''
    Calculate the pre-grasp position before the robot grasp the target object.

    #input
    grasp_pos   : The position of the grasp candidate in cartesian coordinate
    euler       : The orientation angle of the robot hand in pre-grasping configuration
    dist        : Distance of the robot hand to the object in pre-grasping configuration

    #output
    pr_grasp    : Position of the robot hand in pre-grasping configuration in cartesian coordinate
    '''
    phi = euler[0]
    theta = euler[1]
    psi = euler[2]
    rot_x = X_R(phi)
    rot_y = Y_R(theta)
    rot_z = Z_R(psi)
    rel_vect = np.array([0,0,dist])
    pr_grasp = rot_z.dot(rot_y.dot(rot_x.dot(rel_vect)))
    R_camera = Z_R(np.pi)
    pr_grasp = R_camera.dot(pr_grasp)
    pr_grasp = grasp_pos + pr_grasp
    return pr_grasp
        
def pre_grasp_pos_v2(grasp_pos,euler,dist):
    '''
    Calculate the pre-grasp position before the robot grasp the target object.

    #input
    grasp_pos   : The position of the grasp candidate in cartesian coordinate
    euler       : The orientation angle of the robot hand in pre-grasping configuration
    dist        : Distance of the robot hand to the object in pre-grasping configuration

    #output
    pr_grasp    : Position of the robot hand in pre-grasping configuration in cartesian coordinate
    '''

    phi = euler[0]
    theta = euler[1]
    psi = euler[2]
    rot_x = X_R(phi)
    rot_y = Y_R(theta)
    rot_z = Z_R(psi)
    rel_vect = np.array([0,0,dist])
    pr_grasp = rot_x.dot(rot_y.dot(rot_z.dot(rel_vect)))
    R_camera = Z_R(np.pi)
    pr_grasp = R_camera.dot(pr_grasp)
    pr_grasp = grasp_pos + pr_grasp
    return pr_grasp
        




