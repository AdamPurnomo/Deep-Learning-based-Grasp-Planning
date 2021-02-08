# -*- coding: utf-8 -*-
#%%
"""
Created on Wed Jan 15 14:53:27 2020
@author: Adam Syammas Zaki P
"""
import numpy as np
from PIL import Image, ImageDraw
import random
import string
import matplotlib.pyplot as plt
import cv2

def draw_normals(edge_pixels, normals, depthImg, save_dir):
    '''
    Draw normal vector associated with each surface. This function does not return anything.
    Instead, it will save the drawn image directly to the directory.
    
    #input
    edge_pixels : set of edge location in pixel coordinate that represents a surface
                  Shape(n, 2)
    normals     : normal vector associated with each edge pixel
                  Shape (n, 2)
    depthImg    : depth image of the current scene
                  type (uint8)
                  Shape (m,n)
    save_dir    : directory to save the drawn image

    #output
    None
    '''
    depthImg = cv2.cvtColor(depthImg, cv2.COLOR_GRAY2RGB)
    im_pil = Image.fromarray(depthImg)
    draw = ImageDraw.Draw(im_pil)
    i = 0
    while i < len(edge_pixels):
        draw.line(
                (
                        (edge_pixels[i][1], edge_pixels[i][0]),(edge_pixels[i][1] + 5*normals[i][1], edge_pixels[i][0] + 5*normals[i][0])
                        ), 
                fill = (255,0,0)
                )
        i = i+5
    im_pil.save(save_dir)

def draw_grasps (grasps, depthImg, rgb):
    '''
    Draw 2D grasp candidates as a line connecting two grasp contact points. 
    
    #input
    grasps      : Set of 2D grasp candidates 
                  Shape(n, 2, 2)
    depthImg    : depth image of the current scene
                  type (uint8)
                  Shape (m,n)

    #output
    None
    '''
    depthImg = cv2.cvtColor(depthImg, cv2.COLOR_GRAY2RGB)
    im_pil = Image.fromarray(depthImg)
    draw = ImageDraw.Draw(im_pil)
    i = 0
    while(i < len(grasps)):     
        draw.line(
                (
                        (grasps[i,0,0], grasps[i,0,1]), (grasps[i,1,0], grasps[i,1,1])
                        ), 
                fill = rgb
                )
        i = i+1
        
    return np.array(im_pil)

def draw_grasp_representation(grasps, best, ap_vectors, labels, depthImg, data_size, save_dir):
    '''
    Draw grasp rectangle at the best grasp candidate's position
    
    #input
    grasp       : grasp candidates
                  Shape (n,2,2)
    best        : boolean value that indicates whether only the best
                  grasp is drawn or the whole grasp candidates.
    ap_vectors  : 2D projection of grasp approaching pose vectors
    labels      : label of each grasp candidates
    depthImg    : depth image
    data_size   : the size of training data image we want to create.
                  the size will depend on the size of the object
    save_dir    : directory for saving the best grasps

    #output
    dummy       : The whole scene with grasp candidate/s drawn
    '''
    height = data_size[0]
    width = data_size[1]
    img_height, img_width = depthImg.shape
    depthImg = cv2.cvtColor(depthImg, cv2.COLOR_GRAY2RGB)
    dummy = Image.fromarray(depthImg)

    if(best==True):
        iterator = len(grasps)
    else:
        iterator = 1
    
    for i in range(len(grasps)):
        if(best==False):
            grasp = grasps[i]
            ap_vector = ap_vectors[i]
            label = labels[i]
        else:
            grasp = grasps
            ap_vector = ap_vectors
            label = labels   

        #rotating the grasp representation
        center = ((grasp[0] + grasp[1]) / 2).astype(int)
        vect1 = grasp[1] - grasp[0]
        vect2 = grasp[0] - grasp[1]
        angle1 = np.arctan2(vect1[1], vect1[0]) * 180 / np.pi
        angle2 = np.arctan2(vect2[1], vect2[0]) * 180 / np.pi
        angle = np.array([angle1, angle2]) 
        indices = np.argmin(np.abs(angle))
        angle = angle[indices]
        M = cv2.getRotationMatrix2D((center[0], center[1]), -angle, 1.0)

        #defining image size and position
        left = int(center[0] - width/2)
        right = int(center[0] + width/2)
        top = int(center[1] - 20)
        bottom = int(center[1] + 20)
        vertices = ((left,top,1), (left,bottom,1), (right,bottom,1), (right, top,1))
        r_vertices = [M.dot(p) for p in vertices]
        r_vertices = tuple(map(tuple, r_vertices))

        #drawing grasp bounding box
        draw = ImageDraw.Draw(dummy)
        if(label==1):
            draw.polygon(r_vertices, outline = (0,255,0))
            draw.ellipse(((ap_vector[0]-3, ap_vector[1]-3), (ap_vector[0]+3, ap_vector[1]+3)), fill =(0,0,255))
        else:
             draw.polygon(r_vertices, outline = (0,0,255))
        
    #save image
    dummy.save(save_dir)
    return np.array(dummy)


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def data_generator(p_grasps, depthImg, data_size):
    '''
    Generate grasp candidate representation image for training data
    #input
    p_grasps    : projected grasp candidates on the scene image
                  Shape (n, 2, 3)
    p_zvec      : projection of z_vec in the image that represents object orientation
                  Shape (n, 2)
    depthImg    : grayscale depth image (uint8 data)
                  Shape (m, n)
    data_size   : the size of training data image we want to create.
                  the size will depend on the size of the object

    #Output
    None
    img         : set of images which encompass the size of the target object
    bb          : set of images representing the grasp bounding box
                          
    '''
    img = []
    bb = []
    height = data_size[0]
    width = data_size[1]
    r_matrix = []

    depthImg = cv2.cvtColor(depthImg, cv2.COLOR_GRAY2RGB)
    depthImg = cv2.GaussianBlur(depthImg, (11,11), 0)    
    for i in range(len(p_grasps)):
        dummy = Image.fromarray(depthImg)
        
        #rotating the grasp representation
        center = ((p_grasps[i,0] + p_grasps[i,1]) / 2).astype(int)
        vect1 = p_grasps[i,1] - p_grasps[i,0]
        vect2 = p_grasps[i,0] - p_grasps[i,1]
        angle1 = np.arctan2(vect1[1], vect1[0]) * 180 / np.pi
        angle2 = np.arctan2(vect2[1], vect2[0]) * 180 / np.pi
        angle = np.array([angle1, angle2]) 
        indices = np.argmin(np.abs(angle))
        angle = angle[indices]
        dummy = dummy.rotate(angle, center = (center[0], center[1]))
        M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
        r_matrix.append(M)

        #defining image size and position
        left = int(center[0] - width/2)
        right = int(center[0] + width/2)
        top = int(center[1] - height/2)
        bottom = int(center[1] + height/2)
        
        #drawing grasp bounding box
        draw = ImageDraw.Draw(dummy)
        draw.rectangle([(left,center[1]+20), (right-1,center[1]-20)], outline = (255,0,0))
        dummy = dummy.crop((left, top, right, bottom))

        img.append(np.array(dummy))
    return np.asarray(img), r_matrix
    
def save_data(save_dir, label, full_img):
    '''
    Classify grasp candidates based on the defined metric and save the corresponding image

    #input
    save_dir    : base directory where to save the data
    scores      : metrics of grasp candidates
                Shape (n, 4)
    full_img    : image of grasp candidates representation that covers the whole object
                 Shape (n, height, width, 3)

    #output
    nameid      : return name id associated with each grasping candidates
    '''
    data_size = len(full_img)
    nameid = []
    for i in range(data_size):
        if(label[i]==1):
            print('Positive Grasp!')
            root = save_dir[0]
            name = randomString()
            full_arr = root + name +'.png'
            cv2.imwrite(full_arr, full_img[i])
      
        else:
            print('Negative Grasp!')
            root = save_dir[1]
            name = randomString()
            full_arr = root +  name +'.png'
            cv2.imwrite(full_arr, full_img[i])
        
        nameid.append(name)

    return np.asarray(nameid)



        
        
        
        