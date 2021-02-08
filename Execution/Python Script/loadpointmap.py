
# %%
import sys
sys.path.append(r"C:/Users/Adam/Desktop/Controller_Ver.3.8.7/Themes/adam/Execution/Camera Python Binding/x64/Release")

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

    z = np.array(z, dtype='float32')
    y = np.array(y, dtype='float32')
    x = np.array(x, dtype='float32')

    invalid_id = np.where(np.isnan(z))
    mask = np.zeros(z.shape, dtype='uint8')
    mask[invalid_id] = 1

    x = cv2.inpaint(x, mask, 3, cv2.INPAINT_TELEA)
    y = cv2.inpaint(y, mask, 3, cv2.INPAINT_TELEA)
    z = cv2.inpaint(z, mask, 3, cv2.INPAINT_TELEA)
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

def to_cartesian(points, pointmap, double=True):
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
    
    if(double==True):
        center = (0.5*(points[0] + points[1])).astype('int')
        z = estimate_z(points, zmap)
    else:
        center = points.astype('int')
        z = zmap[center[1], center[0]] 
        
    x = xmap[center[1], center[0]]
    y = ymap[center[1], center[0]]
    c = np.array([x, y, z])
    return c

def estimate_z(best_grasp, zmap):

    #parameters for rotation nd cropping
    grasp_vector = best_grasp[1] - best_grasp[0]
    grasp_center = ((best_grasp[1] + best_grasp[0])/2).astype(int)
    angle = np.arctan2(grasp_vector[1], grasp_vector[0]) * 180 / np.pi

    #rotation and cropping
    M = cv2.getRotationMatrix2D((grasp_center[0], grasp_center[1]), angle, 1.0)
    z_rotated = cv2.warpAffine(zmap, M, (zmap.shape[1],zmap.shape[0]))

    contactpoint = np.ones((2,3))
    contactpoint[:,0:2] = best_grasp
    contactpoint = M.dot(contactpoint.T).T.astype(int)

    cp1_idx = np.argmin(contactpoint[:, 0])
    cp2_idx = np.argmax(contactpoint[:, 0])

    cp1 = contactpoint[cp1_idx]
    cp1_left = cp1[0]
    cp1_right = cp1[0] + 40

    cp2 = contactpoint[cp2_idx]
    cp2_left = cp2[0] - 40
    cp2_right = cp2[0]

    cp_top = cp1[1] - 20
    cp_bottom = cp1[1] + 20

    cp1_area = z_rotated[cp_top:cp_bottom, cp1_left:cp1_right]
    cp2_area = z_rotated[cp_top:cp_bottom, cp2_left:cp2_right]
    
    cp1_depth = np.sum(cp1_area)/cp1_area.size
    cp2_depth = np.sum(cp2_area)/cp2_area.size

    z = 0.5*(cp1_depth + cp2_depth)
    return z


