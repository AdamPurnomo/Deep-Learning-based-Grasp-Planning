# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:53:27 2020
@author: Adam Syammas Zaki P
"""
# %%
import numpy as np
from PIL import Image, ImageDraw
import random
import string
import matplotlib.pyplot as plt
import cv2

def draw_normals(edge_pixels, normals, depthImg, save, save_dir):
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
    if(save == True):
            im_pil.save(save_dir)
    return np.array(im_pil)
    

def draw_grasps (grasps, depthImg, rgb, save_dir=None, save=False):
    '''
    Draw 2D grasp candidates as a line connecting two grasp contact points. 
    
    #input
    grasps      : Set of 2D grasp candidates 
                  Shape(n, 2, 2)
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
    while(i < len(grasps)):     
        draw.line(
                (
                        (grasps[i,0,0], grasps[i,0,1]), (grasps[i,1,0], grasps[i,1,1])
                        ), 
                fill = rgb
                )
        i = i+1
    if(save == True):
            im_pil.save(save_dir)
    return np.array(im_pil)


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def draw_best_grapsp(best_grasp, ap_vector, depthImg, data_size, save_dir):
    '''
    Draw grasp rectangle at the best grasp candidate's position
    
    #best_grasp : best grasp candidate
                  Shape (2,2)
    depthImg    : depth image
    data_size   : the size of training data image we want to create.
                  the size will depend on the size of the object
    save_dir    : directory for saving the best grasps
    '''
    height = data_size[0]
    width = data_size[1]
    img_height, img_width = depthImg.shape
    depthImg = cv2.cvtColor(depthImg, cv2.COLOR_GRAY2RGB)
    dummy = Image.fromarray(depthImg)
          
    #rotating the grasp representation
    center = ((best_grasp[0] + best_grasp[1]) / 2).astype(int)
    vect1 = best_grasp[1] - best_grasp[0]
    vect2 = best_grasp[0] - best_grasp[1]
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
    draw.polygon(r_vertices, outline = (0,255,0))
    draw.ellipse(((ap_vector[0]-3, ap_vector[1]-3), (ap_vector[0]+3, ap_vector[1]+3)), fill =(255))
    #save image
    dummy.save(save_dir)


def data_generator(grasps, depthImg, data_size):
    '''
    Generate grasp candidate representation image for training data
    #input
    p_grasps    : projected grasp candidates on the scene image
                  Shape (n, 2, 2)
    depthImg    : grayscale depth image (uint8 data)
                  Shape (m, n)
    data_size   : the size of training data image we want to create.
                  the size will depend on the size of the object

    #Output
    None
    img         : set of images which encompass the size of the target object
    r_matrix    : set of 2d rotation matrix associated with each image of grasp candidates
    '''
    img = []
    r_matrix = []
    height = data_size[0]
    width = data_size[1]
    img_height, img_width = depthImg.shape

    depthImg = cv2.cvtColor(depthImg, cv2.COLOR_GRAY2RGB)
    depthImg = cv2.GaussianBlur(depthImg, (11,11), 0)    
    for i in range(len(grasps)):
        dummy = Image.fromarray(depthImg)
        
        #rotating the grasp representation
        center = ((grasps[i,0] + grasps[i,1]) / 2).astype(int)
        vect1 = grasps[i,1] - grasps[i,0]
        vect2 = grasps[i,0] - grasps[i,1]
        angle1 = np.arctan2(vect1[1], vect1[0]) * 180 / np.pi
        angle2 = np.arctan2(vect2[1], vect2[0]) * 180 / np.pi
        angle = np.array([angle1, angle2]) 
        indices = np.argmin(np.abs(angle))
        angle = angle[indices]
        dummy = dummy.rotate(angle, center = (center[0], center[1]), translate=(img_width/2-center[0], img_height/2-center[1]))
        M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)
        r_matrix.append(M)

        #defining image size and position
        left = int(img_width/2 - width/2)
        right = int(img_width/2 + width/2)
        top = int(img_height/2 - height/2)
        bottom = int(img_height/2 + height/2)

        r_top = int(img_height/2 - 20)
        r_bottom = int(img_height/2 + 20)
        
        #drawing grasp bounding box
        draw = ImageDraw.Draw(dummy)
        draw.rectangle([(left,r_bottom), (right-1,r_top)], outline = (255,0,0))
        dummy = dummy.crop((left, top, right, bottom))

        img.append(np.array(dummy))

    return img, np.array(r_matrix)
            
def save_data(save_dir, full_img, bb_img):
    '''
    Saving data to local directory

    #input 
    save_dir    : base directory to save data
    full_img    : full representation of part with rectangle grasp
    
    #output
    None
    '''
    data_size = len(full_img)
    for i in range(data_size):
        root = save_dir
        name = randomString()
        full_arr = root + r'\full\ ' + name +'.png'
        cv2.imwrite(full_arr, full_img[i])
        bb_arr = root + r'\bounding box\ ' + name +'.png'
        cv2.imwrite(bb_arr, bb_img[i])
            
            


        
        
        
        