## **** Developer/Author: Aditya Jain **** ##
## **** Date Started: 27th March, 2019 **** ##
## **** About: This simulation is being developed to solve the problem of 3D-Bin Packing Problem with Reinforcement Learning ***** #####

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import time
import math
import copy
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import copy

# for having boxes of different colours
boxColorCollection = [[0.137, 0.3529, 0.2863, 1], [0.8745, 0.3176, 0.176, 1], [0.7412, 0.1353, 0.3059, 1], [0.8824, 0.6667, 0.3412, 1], [1, 0.3, 0.3, 1],
[0.3, 0.3, 1, 1], [0.1176, 0.2353, 0.0902, 1], [1, 0, 1, 1], [0, 0, 1, 1], [0.0353, 0.2078, 0.2431, 1], [1, 0, 0, 1], [1, 1, 0, 1],
[0, 1, 0, 1], [0.498, 0.05882, 0.8392, 1], [0.7059, 0.0470, 0.04745, 1], [0.196, 0.5608, 0.549, 1], [0.949, 0.3882, 0.1059, 1], [1, 0.796, 0, 1],
[0.5451, 0.0392, 0.3137, 1], [.9, .9, .9, 1], [0, 0.772, 1, 1], [0.6196, 1, 0.2588, 1], [0.9922, 1, 0.1961, 1]]


class BinPackingEnv():    

    def __init__(self):

        self.gravity = -9.8
        self.state = None
        self.stateWOCrop = None
        self.believedState = None
        self.pixelWidth = 1000
        self.pixelHeight = 1000
        self.camDistance = -3
        self.camTargetPos = [0, 0, 0]
        self.near = 1
        self.far = 1000
        self.fov = 50
        self.boxcount = 0
        self.Y = 0
        self.X = 0
        self.ldcWidth = 0.68
        self.ldcHeight = 0.68
        self.ldcLength = 1.2
        self.slabWidth = 0.02
        self.resp = 0
        self.resw = 0

        # This will be given value in the render function
        self.action_space = None 

        physicsClient = p.connect(p.GUI)

        p.setGravity(0, 0, self.gravity)    # this sets the gravity
        p.setPhysicsEngineParameter(fixedTimeStep=1/60, numSolverIterations=10)

        p.loadSDF('ldc-model-big_center.sdf')
        # p.setRealTimeSimulation(1)

        ######  Adding floor  ########k
        planeId = p.createCollisionShape(p.GEOM_PLANE)
        planeVisualId = p.createVisualShape(p.GEOM_PLANE, rgbaColor=[0.7686, 0.847, 0.933, 1])
        p.createMultiBody(0, planeId, planeVisualId, [0, 0, 0], [0, 0, 0, 1])  

        self.resPix()
        self.resWorld()


    # This functions crops that rgb/depth image to capture only the ldc
    def cropImage(self, image, imageWidth, imageHeight, ldcWidth, ldcLength, fov, camDistance):    
    
        # For along the image width [not being used]
        temp1 = math.tan(math.radians(fov/2))*math.fabs(self.camDistance)
        xunits = 2*temp1
        xres = imageWidth/xunits
    
        # For along the image height
        temp2 = math.tan(math.radians(fov/2))*math.fabs(self.camDistance)
        yunits = 2*temp2
        yres = imageHeight/yunits

        image = np.array(image)
        croppedImage = image[math.floor(imageHeight/2 - yres*ldcWidth/2 + self.slabWidth*yres):math.floor(imageHeight/2 + yres*ldcWidth/2 - self.slabWidth*yres)
                         , math.floor(imageWidth/2 - xres*ldcLength/2 + self.slabWidth*xres):math.floor(imageWidth/2 + xres*ldcLength/2 - self.slabWidth*xres)]
    
        return croppedImage 


    def resPix(self):
        temp = math.tan(math.radians(self.fov/2))*math.fabs(self.camDistance)
        units = 2*temp
        self.resp = units/self.pixelWidth   


    def resWorld(self):
        temp = math.tan(math.radians(self.fov/2))*math.fabs(self.camDistance)
        units = 2*temp
        self.resw = self.pixelWidth/units


    # This gives a random position in the state space in the environment to place the object
    def getRandomAction(self):
        act = [np.random.randint(0,self.X), np.random.randint(0,self.Y)]
        return act


    def reset(self):

        self.state = np.zeros((self.Y, self.X))         # setting the entire state space as zero
        self.believedState = np.zeros((self.Y, self.X))   # setting the believed state space as zero
        self.boxcount = 0
        print("Reset State:", np.shape(self.state))
        return self.state


    # After we have got position to put in image space, we night to find that position in the world frame
    def getPlaceLocation(self, x, y):  

        xpoint = self.resp*x
        ypoint = self.resp*y

        if (xpoint < self.ldcLength/2):
            xpoint = - (self.ldcLength/2 - xpoint)
        else:
            xpoint = xpoint - self.ldcLength/2


        if (ypoint < self.ldcWidth/2):
            ypoint = self.ldcWidth/2 - ypoint
        else:
            ypoint = - (ypoint - self.ldcWidth/2)

        return xpoint, ypoint


    # This function spawns the box inside the ldc given its size, coordinates and mass
    def putBox(self, x, y, length, width, height, mass, xPix, yPix):   # xPix and yPix are the pixel location to be placed

        heightOffset = self.state[yPix, xPix]   

        statetest = np.array(self.state)

        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[float(length)/2, float(width)/2, float(height)/2])
        randomColor = boxColorCollection[self.boxcount]
        visualBoxId = p.createVisualShape(p.GEOM_BOX, rgbaColor=randomColor, halfExtents=[float(length)/2, float(width)/2, float(height)/2])
        p.createMultiBody(mass, colBoxId, visualBoxId, [x + float(length)/2, y - float(width)/2, heightOffset + float(height)/2], [0, 0, 0, 1])
        

    # This function reads the box list file and returns the size and mass of the box
    # As of now reads line-by-line and returns the next box in the list
    def readBoxFile(self): 
        with open("box-list.txt", "r+") as file:    
            for i, line in enumerate(file):
                if i == self.boxcount:

                    line = line.rstrip('\n')
                    line = line.split(' ')
                    line = list(filter(None, line))

                    self.boxcount += 1                  
                    return [float(line[0]), float(line[1]), float(line[2]), float(line[3])]


    # def step(self, action):
    def step(self):

        # as of now taking random action
        self.action_space = self.getRandomAction()

        action_space = self.action_space    # getting the action here i.e. where to keep the object
        
        box_info = self.readBoxFile()       # reading the box dimenions

        # cv2.imshow("Real1 State Space", self.state)  

        xplace, yplace = self.getPlaceLocation(action_space[0], action_space[1])
        # print("Place Location:", xplace, yplace)

        self.render()

        self.putBox(xplace, yplace, box_info[0], box_info[1], box_info[2], box_info[3], action_space[0], action_space[1])

        for i in range(100):
            p.stepSimulation()
        
        self.render()
        print("Max Height: ", np.max(self.state))

        reward = 0   
        done = False  
        ###############################

        return np.array(self.state), reward, done, {} 


    def render(self, mode='human', close=False):

        ## setting the camera parameter     
        pitch = 0
        roll=0
        upAxisIndex = 1
        yaw = 0
        aspect = self.pixelWidth / self.pixelHeight
        
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(self.camTargetPos, self.camDistance, yaw, pitch, roll, upAxisIndex)
        projectionMatrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)
        
        # p.stepSimulation()

        # GUI
        img_arr = p.getCameraImage(self.pixelWidth, self.pixelHeight, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)       


        w = img_arr[0]   #width of the image, in pixels
        h = img_arr[1]   #height of the image, in pixels
        rgb = img_arr[2] #color data RGB
        dep = img_arr[3] #depth data   

        rgb_opengl = np.reshape(img_arr[2], (self.pixelHeight, self.pixelWidth, 4))*1./255.
        # cv2.imshow("cropped Image RGB", rgb_opengl)

        depth_buffer_opengl = np.reshape(img_arr[3], [self.pixelHeight, self.pixelWidth])
        # cv2.imshow("cropped Image Disparity", croppedImage2)

        croppedImage2 = self.cropImage(depth_buffer_opengl, self.pixelWidth, self.pixelHeight, self.ldcWidth, self.ldcLength, self.fov, self.camDistance)
        size = np.shape(croppedImage2)
        self.Y = size[0]
        self.X = size[1]

        depth_opengl = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer_opengl)

        # while(1):
            # time.sleep(0.01)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        actual_state_space = math.fabs(self.camDistance) - depth_opengl

        self.state = actual_state_space 
        
        return self.state 


#### Main Function ####
simrun = BinPackingEnv()
num_boxes = 4  ## can vary this according to your need

simrun.reset()
simrun.render()

for i in range(num_boxes):
    simrun.step()