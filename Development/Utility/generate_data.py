#%%
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import sys
import antipodal_sampler as ap
import img_processor as ip  
import json

par_dir = r'..\..\Simulation\source code\JSON File\parameters.json'

with open(par_dir) as f:
    parameters = json.load(f)

def generate_data(part_name, rawdata_num):

    #setting directory
    save_dir = r'../Training Data/Real Data/'+part_name+r'/un-annotated/'
    pointmap_dir = r'../Point Map Data/' + part_name
    
    #setting parameters
    grasp_range = parameters[part_name][0:2]
    data_size = parameters[part_name][4:6]
    for i in range(1,rawdata_num):
        pm_dir = pointmap_dir + r'/pointmap' + str(i) + '.npy'
        #loading pointmap from temporary folder
        pointmap = np.load(pm_dir)
        z = pointmap
        depthMap = z.reshape((1024,1280))
        depthMapROI = depthMap[230:730, 200:970]

        #generating depth image from depth map
        depthImg = cv2.normalize(depthMapROI,
                        dst=None,
                        alpha=0,
                        beta=255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8UC1)
        depthImg = 255-depthImg

        #filling missing values from depth map
        mask = np.zeros(depthImg.shape).astype('uint8')
        indices = np.where(depthImg == 255)
        mask[indices] = 1
        depthImg = cv2.inpaint(depthImg, mask, 3, cv2.INPAINT_TELEA)

        #computing edge image
        blur = cv2.GaussianBlur(depthImg, (7,7), 0)
        edgeImg = cv2.Canny(blur, 5, 60)

        #grasps sampling
        normals, edgepx = ap.normals(depthImg, edgeImg)
        grasps, normals = ap.antipodal_sampler(edgepx, normals, grasp_range)

        #visualize grasp
        grasp_img = ip.draw_grasps(grasps, depthImg,(0,0,255))

        #generating grasp image representations
        data, _ = ip.data_generator(grasps, depthImg, data_size)

        ip.save_data(save_dir,data)


generate_data('hv6',63)
