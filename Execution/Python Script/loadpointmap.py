
# %%
import sys
sys.path.append(r"../Camera Python Binding/x64/Release")

import numpy as np 
import cv2
import getpointmap


def load_pointmap(img_size, roi):
    '''
    Load pointmap stored in temporary data directory in the form of txt file
    #input
    file_dir    : directory where the file is saved
    img_size    : image size to reshape the pointmap
    roi         : region of interest to specify 
                  which part should be processed
                  [top, bottom, left, right]
    None
    #Output
    x   : map of x coordinates of the scene
    y   : map of y coordinates of the scene
    z   : map of z corrdinates of the scne
    '''
    pointmap = getpointmap.getpointmap()
    pointmap = np.array(pointmap)
    xindices = np.arange(0, len(pointmap), 3)
    yindices = xindices + 1
    zindices = yindices + 1
    
    x = pointmap[xindices]
    y = pointmap[yindices]
    z = pointmap[zindices]

    x = x.reshape((img_size[0], img_size[1]))
    y = y.reshape((img_size[0], img_size[1]))
    z = z.reshape((img_size[0], img_size[1]))

    x = x[roi[0]:roi[1], roi[2]:roi[3]]
    y = y[roi[0]:roi[1], roi[2]:roi[3]]
    z = z[roi[0]:roi[1], roi[2]:roi[3]]
    return x, y, z

def to_depthImg(depthmap):
    '''
    convert depth map to depth image
    #input
    depthmap    : depthmap taken from camera
                  Shape(height, width)
    
    #output
    depthImg    : Processed depth image
                 uint8 data type
                 Shape (height, width
    '''
    depthImg = 255 - cv2.normalize(depthmap, 
                            dst=None, 
                            alpha=0, 
                            beta=255, 
                            norm_type=cv2.NORM_MINMAX, 
                            dtype=cv2.CV_8UC1)
    invalid_id = np.where(depthImg == 255)
    mask = np.zeros(depthImg.shape, dtype = 'uint8')
    mask[invalid_id] = 255
    depthImg = cv2.inpaint(depthImg,mask,3,cv2.INPAINT_TELEA)

    return depthImg

def to_cartesian(best_grasp, pointmap):
    '''
    Transform the location of best grasp candidates from pixel coordinate to cartesian coordinate
    
    #input
    best_grasp  : best grasp candidates in pixel coordinate
    pointmap    : pointmap

    #output
    c : best grasp candidates in cartesian

    '''
    xmap = pointmap[0]
    ymap = pointmap[1]
    zmap = pointmap[2]

    center = (0.5*(best_grasp[0] + best_grasp[1])).astype('int')

    x = xmap[center[1], center[0]]
    y = ymap[center[1], center[0]]
    z = zmap[center[1], center[0]]
    
    c = np.array([x, y, z])
    return c




