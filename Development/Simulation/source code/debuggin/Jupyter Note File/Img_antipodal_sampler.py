# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:23:27 2019

@author: Adam Syammas Zaki P
"""

friction_coef = 0.5

""" 
There is something important to note. Majority of image processor library are using (x,y)
coordinate system, while numpy works like a matrix where the coordinate of pixel is written in (row, colum)
or in another word (y,x). That is why on line 133 in this code, I deliberately exchange the indices of the grasp
candidates so that it is represented in the form of (x,y)
"""


import numpy as np
from scipy import ndimage as nd
#import matplotlib.pyplot as plt
#import cv2
from PIL import Image, ImageDraw
import scipy.spatial.distance as ssd
#import time

def draw_normals(edge_pixels, normals, depth_im, save_arr):
    depth_im = Image.open(depth_im)
    draw = ImageDraw.Draw(depth_im)
    i = 0
    while i < len(edge_pixels):
        draw.line(
                (
                        (edge_pixels[i][1], edge_pixels[i][0]),(edge_pixels[i][1] + 5*normals[i][1], edge_pixels[i][0] + 5*normals[i][0])
                        ), 
                fill = (255,0,0)
                )
        i = i+5
    depth_im.save(save_arr)

def draw_grasps (grasps, depth_im, save_arr):
    depth_im = Image.open(depth_im)
    draw = ImageDraw.Draw(depth_im)
    i = 0
    while(i < len(grasps)):
        draw.line(
                (
                        (grasps[i,0,1], grasps[i,0,0]), (grasps[i,1,1], grasps[i,1,0])
                        ), 
                fill = (255,0,0)
                )
        i = i+1
        
    depth_im.save(save_arr)

def draw_com(com, segimg, save_arr):
    img = Image.open(segimg)
    draw = ImageDraw.Draw(img)
    draw.line(((com[0] - 2, com[1]), (com[0] + 2, com[1])), fill = (255, 0, 0))
    draw.line(((com[0], com[1] - 2), (com[0], com[1] + 2)), fill = (255, 0, 0))
    #draw.point(com, fill = (255, 0, 0))
    img.save(save_arr)
    
def normals(depth, edge_im):
    edge_pixels = np.where(edge_im != [0])
    edge_pixels = np.c_[edge_pixels[0], edge_pixels[1]]
    depth = np.float64(depth) #this is very important to note. Gradient image will only work if the data is in the form of float64
    sx = nd.sobel(depth, axis=0, mode= 'constant') #gradient in x direction
    sy = nd.sobel(depth, axis=1, mode='constant') #gradient in y direction
    
    normals = np.zeros((len(edge_pixels),2))
    
    #gradient of images always point out to normal of the edges
    for i, pix in enumerate(edge_pixels):
        dx = sx[pix[0], pix[1]]
        dy = sy[pix[0], pix[1]]
        normal_vec = np.array([-dx,-dy], dtype=float) #input normal vector of each egde pixel
        if np.linalg.norm(normal_vec) == 0:
            normal_vec = np.array([1,0])
        normal_vec = normal_vec / np.linalg.norm(normal_vec) #normalize the normal vector 
        normals[i, :] = normal_vec #append normal vector to an array so that it has the same index as the edge pixel
    return normals, edge_pixels

def antipodal_sampler (edge_pixels, normals, grasp_range):
    min_dist = grasp_range[0]
    max_dist = grasp_range[1]
    
    dists = ssd.squareform(ssd.pdist(edge_pixels))
    normals_ip = normals.dot(normals.T)
    
    valid_pairs = np.where(
            (normals_ip < -0.9)&
            (dists < max_dist) &
            (dists > min_dist)
            )

    valid_pairs = np.c_[valid_pairs[0], valid_pairs[1]]

    contact_points1 = edge_pixels[valid_pairs[:,0], :]
    contact_points2 = edge_pixels[valid_pairs[:,1], :]
    normals_points1 = normals[valid_pairs[:,0], :]
    normals_points2 = normals[valid_pairs[:,1], :]
    
    v = contact_points1 - contact_points2
    v_norm = np.linalg.norm(v, axis=1)
    v = v / np.tile(v_norm[:, np.newaxis], [1, 2])
    ip1 = np.sum(normals_points1 * v, axis=1)
    ip2 = np.sum(normals_points2 * (-v), axis=1)
    antipodal_indices = np.where((ip1<-np.cos(np.arctan(friction_coef)))&(ip2<-np.cos(np.arctan(friction_coef))))[0]
    grasp_indices = np.random.choice(antipodal_indices, size = antipodal_indices.shape[0], replace = False)

    k = 0
    grasps = []
    center = []
    normals = []
    while k < grasp_indices.shape[0] and len(grasps) < 200:
        p1 = contact_points1[grasp_indices[k],:]
        p2 = contact_points2[grasp_indices[k],:]
        c = np.array([p1 + ((p2 - p1) * 1 / 5), 
                      p1 + ((p2 - p1) * 2 / 5),
                      p1 + ((p2 - p1) * 3 / 5),
                      p1 + ((p2 - p1) * 4 / 5)])
        n1 = normals_points1[grasp_indices[k],:]
        n2 = normals_points2[grasp_indices[k],:]
        
        grasps.append([p1, p2])
        normals.append([n1, n2])
        center.append(c)
        k = k+1
        
    grasps = np.asarray(grasps)
    normals = np.asarray(normals)
        
    g_zeros = np.zeros(grasps.shape)
    if(grasps.size !=0):
        g_zeros[:,:,0] = grasps[:,:,1]
        g_zeros[:,:,1] = grasps[:,:,0]
        grasps = g_zeros.astype(int)
    
    return grasps, normals, np.int_(center)



#depth_image = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\debug\depth.png'
#edge_image = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\debug\edge.png'

#image = cv2.imread(depth_image, cv2.IMREAD_GRAYSCALE)
#edge_im = cv2.imread(edge_image)

#normal_px, edge_px = normals(image, edge_im)
#draw_normals(edge_px, normal_px, depth_image, r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\debug\normals.png')
#grasps_sg, normals_sg, center = antipodal_sampler(edge_px, normal_px)

r"""
### Grasps Sampling with single Image
time1 = time.clock()
time2 = time.clock()
depth_image = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\depth.png'
edge_im = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\edge.png'
new_edge_im = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\newedge.png'
image = cv2.imread(depth_image, cv2.IMREAD_GRAYSCALE)

edge_im = cv2.imread(edge_im)
new_edge_im = cv2.imread(new_edge_im)

normal_px, edge_px = normals(image, edge_im)
nwnormal_px, nwedge_px = normals(image, new_edge_im)

save_arr1 = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\normals.png'
save_arr2 = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\grasp.png'
save_arr3 = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\new_grasp.png'


grasps_sg, normals_sg, center = antipodal_sampler(edge_px, normal_px)
nwgrasps_sg, nwnormals_sg, center = antipodal_sampler(nwedge_px, nwnormal_px)

time3 = time.clock()
#draw_normals(edge_px, normal_px, depth, save_arr1)
draw_image(grasps_sg, depth_image, save_arr2)
draw_image(nwgrasps_sg, depth_image, save_arr3)
print('Potential grasps generation:     %.3f s'%(time3-time2))
"""






r"""
### simple rectangle
simple_im =  r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\simple.png'
sm_im = cv2.imread(simple_im, cv2.IMREAD_GRAYSCALE)
filt = cv2.GaussianBlur(sm_im, (5,5), 0)
sm_edge_im = cv2.Canny(filt, 1, 6)
cv2.imwrite(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\simple_debug_edge.png', sm_edge_im)

normals_pm, edge_pm = normals(sm_im, sm_edge_im)
cv2.imwrite(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\simple_debug_edge.png', sm_edge_im)
save_arr1 = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\sm_normal_debug.png'


save_arr2 = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\sm_grasp_debug.png'
depth = Image.open(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\simple.png')
grasps_sm, normals_sm = antipodal_sampler(edge_pm, normals_pm)
#draw_normals(edge_pm, normals_pm, depth, save_arr1)
draw_image(grasps_sm, depth, save_arr2)
###
"""
#sx = nd.sobel(image, axis=0, mode= 'constant')
#sy = nd.sobel(image, axis=1, mode='constant')
#sobel = np.hypot(sx, sy)
#np.savetxt('sobel.csv',sobel ,delimiter=',')
#img_copy = np.uint8(img)
#edge_im = cv2.Canny(img_copy, 1, 5)
#np.savetxt('canny.csv', edge_im, delimiter=',')

r""" 
### Generate simple rectangle
sm_im = np.zeros((100,100))
sm_im[30:-30, 20:-20] = 1
sm_im = nd.rotate(sm_im, 25, mode='constant')
#sm_im_copy = np.uint8(sm_im)

simple_im =  r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\debug\simple.png'
plt.imsave(simple_im, sm_im,  cmap='gray_r')
depth = cv2.imread(simple_im, cv2.IMREAD_GRAYSCALE)
depth = np.uint8(depth)
filt = cv2.GaussianBlur(depth, (5,5), 0)

sm_edge_im = cv2.Canny(filt, 1, 6)
cv2.imwrite(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\debug\simple_debug_edge.png', sm_edge_im)
normal, edge = normals(depth, sm_edge_im)
grasps, normals, center = antipodal_sampler(edge, normal)
save_arr = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\debug\grasp.png'
draw_image(grasps, simple_im,save_arr)
"""



