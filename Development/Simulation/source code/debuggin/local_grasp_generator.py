"""
Created on Tue Nov 12 16:07:39 2019
@author: Adam Syammas Zaki P
"""
# %%
import numpy as np
import scipy as sp
import pybullet as pb
import pybullet_data
import cv2
import time
import load_object as lo
import transformation as t
import antipodal_sampler as ap
import ray_tracing as rt
import wrench_space_analysis as wsa
import img_processor as ip 
import metrics as m 

from skimage.morphology import square
from skimage.morphology import closing


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

parameters = {'hv18': [32, 40, 0.02, 0.025, 220, 110, 8000,
                       'cylindrical',
                       r'..\..\..\data\model\hv18_2.obj', 
                       r'..\..\..\data\model\hv18_2.stl',
                       r'..\..\..\Simulation\Image\Training Data\Classification\hv18\Positive\data\ ',
                       r'..\..\..\Simulation\Image\Training Data\Classification\hv18\Negative\data\ ',
                       r'..\..\..\Simulation\Image\Training Data\Classification\hv18\Positive\mask\ ',
                       r'..\..\..\Simulation\Image\Training Data\Classification\hv18\Negative\mask\ '], 
              'hv8':[38, 44, 0.09, 0.11, 220, 220, 23000,
                     'cicular',
                     r'..\..\..\data\model\hv8.obj', 
                     r'..\..\..\data\model\hv8.stl',
                     r'..\..\..\Simulation\Image\Training Data\Classification\hv8\Positive\data\ ',
                     r'..\..\..\Simulation\Image\Training Data\Classification\hv8\Negative\data\ ',
                     r'..\..\..\Simulation\Image\Training Data\Classification\hv8\Positive\mask\ ',
                     r'..\..\..\Simulation\Image\Training Data\Classification\hv8\Negative\mask\ ']}

        
def main(part_name):
    #initializing parameters 
    part_parameters = parameters[part_name]
    grasp_range = [part_parameters[0], part_parameters[1]]    
    image_size = [part_parameters[4], part_parameters[5]]
    nonoccluded_pixel = part_parameters[6]
    part_dir = [part_parameters[8], part_parameters[9]]
    save_dir = [part_parameters[10], part_parameters[11], part_parameters[12], part_parameters[13]]


    local_com = np.array([0,0,0])
    far = 8.7
    aspect = 1.25
    near = 7.3
    fov = 37
    img_size = [1024, 1280]
    renderingParameters = [far, near, aspect, fov, img_size]
    
    #initializing simulation env
    physicsClient = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    container = lo.load_container()
    
    #All the unit is in dm because the size of model is enlarged 10 times
    pb.setGravity(0,0,-98) 
    pb.loadURDF('plane.urdf')
    num_parts = np.random.randint(2, 25)
   
    #loading parts and simulating
    position = [0,0,2]
    orientation = [0, 0, 0]
    partID = lo.load_parts(position, orientation, part_dir)

    for i in range (1000):
        pb.stepSimulation(physicsClient)
        time.sleep(1./240.)

    #image rendering   
    viewMatrix = pb.computeViewMatrix(
        cameraEyePosition=[0, 0, 9.1],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, 1, 0])
            
    projectionMatrix = pb.computeProjectionMatrixFOV(
                fov=fov,
                aspect=aspect,
                nearVal=near,
                farVal=far)
            
    width, height, rgbImg, depthBuff, segImg = pb.getCameraImage(
                width=img_size[1], 
                height=img_size[0],
                viewMatrix=viewMatrix,
                projectionMatrix=projectionMatrix)
    
    viewMatrix = np.reshape(viewMatrix, (4,4), 'F')
    projectionMatrix = np.reshape(projectionMatrix, (4,4), 'F')
    renderingMatrices = [viewMatrix, projectionMatrix]


    #retrieving position and orientation of each object in the bin    
    position, o = pb.getBasePositionAndOrientation(partID)
    orientation = np.reshape(pb.getMatrixFromQuaternion(o), (3,3))
    
    #Grasp sampling from depth image and edge image
    depthImg = cv2.normalize(depthBuff, 
                            dst=None, 
                            alpha=0, 
                            beta=255, 
                            norm_type=cv2.NORM_MINMAX, 
                            dtype=cv2.CV_8UC1)
    segGrey = cv2.normalize(segImg, 
                            dst=None, 
                            alpha=0, 
                            beta=255, 
                            norm_type=cv2.NORM_MINMAX, 
                            dtype=cv2.CV_8UC1)
    segGrey = closing(segGrey, square(5))

    depthImg = cv2.GaussianBlur(depthImg, (11,11), 0)
    edgeImg = cv2.Canny(segGrey, 5, 40)
    normal_px, edge_px = ap.normals(depthImg, edgeImg)
    grasps, grasp_normals = ap.antipodal_sampler(edge_px, normal_px, grasp_range)

    #draw grasps on depth image
    grasp_dir = r'..\..\Image\Temporary\grasps.png'
    ap.draw_grasps(grasps, depthImg, grasp_dir)
    cv2.imwrite(r'..\..\Image\Temporary\seggrey.png', segGrey)
    cv2.imwrite(r'..\..\Image\Temporary\edge.png', edgeImg)

    #projecting grasp candidates to 3d cartesian coordinate
    depthMap = t.to_depthMap(depthBuff, renderingParameters) 
    p_grasps = t.grasps_projection(grasps, 
                                 depthMap,
                                 renderingParameters, 
                                 renderingMatrices) 
  
    #correction of grasp projections
    p_grasps, Tmatrix = t.orientation_correction(p_grasps,
                                                        segImg,
                                                        position,
                                                        orientation)

    #Transformation of grasp candidates from universal coordinate to object local coordinate
    local_grasps = t.grasp_local_transformation(Tmatrix, p_grasps) 

    #Ray tracing to obtain normal surface of each grasp candidates
    lx_grasps, normal_grasps, invalid_indexes = rt.ray_tracing(local_grasps, part_dir[1]) 

    
    #pb.disconnect()
    '''
    #Wrench Space Analysis
    grasp_scores, l_scores, invalid_indexes = wsa.evaluate_grasps(lx_grasps, normal_grasps, local_com)

    #grasps pruning
    grasps = np.delete(grasps, invalid_indexes, axis = 0)
    lx_grasps = np.delete(lx_grasps, invalid_indexes, axis = 0)
    normal_grasps = np.delete(normal_grasps, invalid_indexes, axis = 0)
    '''

    #force closure
    fc_scores = m.force_closure(lx_grasps, normal_grasps, 0.5)

    #grasps visualization
    u_grasps = t.grasp_univ_transformation(Tmatrix, lx_grasps)
    t.draw_realgrasps(u_grasps, fc_scores)

    #saving grasp candidates and scores data
    '''
    np.save(r'./local grasps/'+part_name+r'/local_grasps.npy', lx_grasps)
    np.save(r'./local grasps/'+part_name+r'/grasp_scores.npy', grasp_scores)
    
    grasp_img = ip.data_generator(grasps,
                                lx_grasps,
                                depthImg,
                                grasp_scores,
                                image_size,
                                save_dir)
    '''

    return lx_grasps, fc_scores, normal_grasps

lx_grasps, fc_scores, normal_grasps = main('hv18')
    
