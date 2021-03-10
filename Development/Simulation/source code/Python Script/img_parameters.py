#%%
import matplotlib.pyplot as plt 
import numpy as np
import pybullet as pb
import pybullet_data
import trimesh as tr
import random as r
import time
import load_object as lo
import json

"""This is the simulation file that needs to be run in prior of generating the synthethic data. 
This simulation will calculate the appropriate image size and max-min grasp distance in term of pixel 
for each object category. However, for a very concave object like hv18, this calculation may not be accurate.

It should be noted that the dimension between trimesh environment and pybullet environment is different."""


with open(r'..\JSON File\parameters.json') as f:
    parameters = json.load(f)

pi = np.pi/180
draw = 1
printtext = 0

def calculate_param(part_name,scale):
    '''
    Calculate parameters of images for training data of each particular object type

    #input
    part_name   : The name of the object type
    scale       : Scale differences between trimesh and pybullet environment

    #output
    None. It will print the rough estimate of the needed parameters to create images.
    '''
    part_parameters = parameters[part_name]
    part_dir = [part_parameters[7], part_parameters[8]]

    physicsClient = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = pb.loadURDF('plane.urdf')
    pb.setGravity(0,0,-98)

    far = 8.7
    aspect = 1.25
    near = 7.3
    fov = 37
    cam_pos = [0,0,9.1]
    img_size = [1024, 1280]
    renderingParameters = [far, near, aspect, fov, img_size]

    mesh = tr.load_mesh(part_dir[1], process = False)
    size = scale*mesh.bounding_box.primitive.extents 

    position = [0,0,cam_pos[2] - (far+near)/2]
    orientation = [0,0,0]
    partID = lo.load_parts(position, orientation, part_dir,scale)
    container = lo.load_container()

    for i in range (1000):
        pb.stepSimulation(physicsClient)
        time.sleep(1./240.)
    

    viewMatrix = pb.computeViewMatrix(
                cameraEyePosition=cam_pos,
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[0, 1, 0])
            
    projectionMatrix = pb.computeProjectionMatrixFOV(
                fov=fov,
                aspect=aspect,
                nearVal=near,
                farVal=far)
            
    width, height, rgbImg, depthbuff, segImg = pb.getCameraImage(
                width=img_size[1], 
                height=img_size[0],
                viewMatrix=viewMatrix,
                projectionMatrix=projectionMatrix)

    position, orientation = pb.getBasePositionAndOrientation(partID)
    total_pixel  = np.sum(segImg == partID)



    print('Calculating Parameters...')


    v1 = np.array([size[0], size[1], cam_pos[2]-near, 1])
    v2 = np.array([size[0], size[1], cam_pos[2]-far, 1])

    V = np.reshape(viewMatrix, (4,4), order = 'F')
    P = np.reshape(projectionMatrix, (4,4), order = 'F')

    viewpos1 = np.dot(V, v1)
    viewpos2 = np.dot(V, v2)

    clippos1 = np.dot(P, viewpos1)
    clippos2 = np.dot(P, viewpos2)

    normalpos1 = (clippos1 / clippos1[3])[0:3]
    normalpos2 = (clippos2 / clippos2[3])[0:3]

    row_size, column_size = img_size

    column1 = (normalpos1[0] + 1)*(column_size - 1) / 2
    row1 = (1 - normalpos1[1])*(row_size-1) / 2

    column2 = (normalpos2[0] + 1)*(column_size - 1) / 2
    row2 = (1 - normalpos2[1])*(row_size - 1) / 2

    #dimentsion in pixel
    dim1 = [abs(column1 - column_size / 2), abs(row1 - row_size / 2)]
    dim2 = [abs(column2 - column_size / 2), abs(row2 - row_size / 2)]

    print('Approximate Image Size for Training Data : ', dim1, '\n', 
          'Approximate Image Size for Training Data : ', dim2) 
    print('Total Number of pixel : ', total_pixel)
    
    pb.disconnect()
              
              



if __name__ == "__main__":
    calculate_param('hv6', 10)
