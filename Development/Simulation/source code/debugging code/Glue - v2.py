"""
Created on Tue Nov 12 16:07:39 2019
@author: Adam Syammas Zaki P
"""
# %%
import numpy as np
import scipy as sp
import pybullet as pb
import pybullet_data
import matplotlib.pyplot as plt
import cv2
import random as r
import time
from skimage.morphology import closing, square
import Simulation as s
import Pixel_realworld_transformation as pr
import Img_antipodal_sampler as spl
import ray_tracing as rt
import Wrench_Space_Analysis as wsa
import img_processor as ip
import occlusion_rate as ocr
import threed_antipodal_sampler as tdspl
import collision_estimation as cst

pi = np.pi/180

"""
Note on parameters dictionary. This parameters describe the parameters belong to each specific part. 
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

parameters = {'hv18': [8, 12, 0.02, 0.025, 60, 24, 500,
                       'cylindrical',
                       r'..\..\..\data\model\hv18_2.obj', 
                       r'..\..\..\data\model\hv18.stl',
                       r'..\..\..\Simulation\Image\Training Data\Classification\hv18\Positive\ ',
                       r'..\..\..\Simulation\Image\Training Data\Classification\hv18\Negative\ '], 
              'hv8':[38, 44, 0.09, 0.11, 60, 60, 1343,
                     'cicular',
                     r'..\..\..\data\model\hv8.obj', 
                     r'..\..\..\data\model\hv8.stl',
                     r'..\..\..\Simulation\Image\Training Data\Classification\hv8\Positive\ ',
                     r'..\..\..\Simulation\Image\Training Data\Classification\hv8\Negative\ ']}

        
def main(part_name):
    #initializing parameters 
    part_parameters = parameters[part_name]
    grasp_range = [part_parameters[0], part_parameters[1]]    
    image_size = [part_parameters[4], part_parameters[5]]
    nonoccluded_pixel = part_parameters[6]
    part_dir = [part_parameters[8], part_parameters[9]]
    save_dir = [part_parameters[10], part_parameters[11]]

    local_com = np.array([0,0,0])
    far = 6.1
    aspect = 1
    near = 0.1
    fov = 45.0
    img_size = 224
    renderingParameters = [far, near, aspect, fov, img_size]
    
    #initializing simulation env
    physicsClient = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    #All the unit is in dm because the size of model is enlarged 10 times
    pb.setGravity(0,0,-98) 
    pb.loadURDF('plane.urdf')
    partID = []
    num_parts = np.random.randint(2, 25)
   
    #loading parts and simulating
    for i in range (1):
        position = [0,0,1]
        orientation = [0, 0, 0]
        partID.append(s.load_parts(position, orientation, part_dir))
    partID = np.asarray(partID)

    for i in range (1000):
        pb.stepSimulation(physicsClient)
        time.sleep(1./240.)

    #image rendering   
    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=[0, 0, 6],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0])
            
    projectionMatrix = pb.computeProjectionMatrixFOV(
                fov=fov,
                aspect=aspect,
                nearVal=near,
                farVal=far)
            
    width, height, rgbImg, depthBuff, segImg = pb.getCameraImage(
                width=img_size, 
                height=img_size,
                viewMatrix=viewMatrix,
                projectionMatrix=projectionMatrix)
    
    viewMatrix = np.reshape(viewMatrix, (4,4), 'F')
    projectionMatrix = np.reshape(projectionMatrix, (4,4), 'F')
    renderingMatrices = [viewMatrix, projectionMatrix]

    #directory to save files
    depthImg_dir = r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\Simulation\Image\Temporary\depth.png'
    segImg_dir = r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\Simulation\Image\Temporary\seg.png'
    edgeImg_dir = r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\Simulation\Image\Temporary\edge.png'
    graspImg_dir = r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\Simulation\Image\Temporary\grasps.png'
    

    #retrieving position and orientation of each object in the bin    
    position = []
    orientation = []
    for ID in partID:
        p, o = pb.getBasePositionAndOrientation(ID)
        o = np.reshape(pb.getMatrixFromQuaternion(o), (3,3))
        position.append(p)
        orientation.append(o)
    
    #Grasp sampling from depth image and edge image
    depthImg = cv2.normalize(depthBuff, 
                            dst=None, 
                            alpha=0, 
                            beta=255, 
                            norm_type=cv2.NORM_MINMAX, 
                            dtype=cv2.CV_8UC1)
    blur = cv2.GaussianBlur(depthImg, (11,11), 0)
    np.save('depth_buffer.npy', depthBuff)
    edgeImg = cv2.Canny(blur, 5, 20)
    normal_px, edge_px = spl.normals(depthImg, edgeImg)
    grasps, grasp_normals, centers = spl.antipodal_sampler(edge_px, normal_px, grasp_range)
    np.save('grasps.npy', grasps)
    spl.draw_grasps(grasps, depthImg, graspImg_dir)
    #saving images
    cv2.imwrite(depthImg_dir, depthImg)
    cv2.imwrite(edgeImg_dir, edgeImg)

    #projecting grasp candidates to 3d cartesiam coordinate
    depthMap = pr.to_depthMap(depthBuff, renderingParameters) 
    p_grasps = pr.grasps_projection(grasps, 
                                 depthMap,
                                 renderingParameters, 
                                 renderingMatrices) 
  
    #correction of grasp projections
    p_grasps, Tset, graspID = pr.orientation_correction(p_grasps,
                                                        centers,
                                                        segImg,
                                                        partID,
                                                        position,
                                                        orientation)

    #Grasp candidates pruning
    nonpart_indexes, _ = np.where(graspID<1)
    graspID = np.delete(graspID,nonpart_indexes, axis = 0)
    grasps = np.delete(grasps, nonpart_indexes, axis = 0)
    p_grasps = np.delete(p_grasps, nonpart_indexes, axis = 0)

    #Index of graspID in partID
    graspID_index = np.searchsorted(partID, graspID)

    #Transformation of grasp candidates from universal coordinate to object local coordinate
    local_grasps = pr.grasp_local_transformation(Tset, p_grasps, graspID_index) 
    np.save('local_grasps.npy', local_grasps)

    #Ray tracing to obtain normal surface of each grasp candidates
    lx_grasps, normal_grasps, invalid_indexes = rt.ray_tracing(local_grasps, part_dir[1]) 
    graspID_index = np.delete(graspID_index, invalid_indexes, axis = 0)

    
    #pb.disconnect()
    #Wrench Space Analysis
    grasp_scores, invalid_indexes = wsa.evaluate_grasps(lx_grasps, normal_grasps, local_com)

    #grasps pruning
    grasps = np.delete(grasps, invalid_indexes, axis = 0)
    lx_grasps = np.delete(lx_grasps, invalid_indexes, axis = 0)
    graspID_index = np.delete(graspID_index, invalid_indexes, axis = 0)

    #grasps visualization
    u_grasps = pr.grasp_univ_transformation(Tset, lx_grasps, graspID_index)
    pr.draw_realgrasps(u_grasps, grasp_scores)
    print(u_grasps.shape)

    #ocr_scores = ocr.occlusion_rate(segImg, ID, grasps, image_size, nonoccluded_pixel)
    #collision_scores = cst.collision_est(depthImg, segImg, ID, grasps, image_size)
         
    #scores = np.array([grasp_scores, loc_scores, ocr_scores, collision_scores])
        #ip.img_generator(grasps, depth_loc, scores, image_size, save_dir)
    
 
 
   
    #return segImg, ID, grasps, depthImg

#iterate the program
# %%
for i in range(1):
    main('hv8')
    
