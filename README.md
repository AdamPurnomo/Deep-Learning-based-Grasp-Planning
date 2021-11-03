# Deep Learning-based 6-DoF Grasp Estimation for Industrial Bin-Picking

## Overview
This project focus on the problem of the 6-DoF parallel-jaw grasp for bin-picking in an industrial setting. We explore a method to estimate 6-DoF
grasp from bin-picking scenes in an industrial setting using a convolutional neural network (CNN) without having to estimate the 6D pose of the target object. We gained inspiration from the idea that the robot does not need to know exactly the
precise 6D pose of the target object to get a reasonably good grasp. A rough estimate on how to approach the object is often more than sufficient to grasp the target object. We use what we call grasp approaching pose vector which determines from which direction the robot gripper should approach the target object in Cartesian space. The network evaluates the grasp candidates represented as grasp rectangle taken from a single depth image and outputs the 2D projection of grasp approaching pose vector at once. This 2D projection can later be converted back to a 3D vector with the knowledge of the camera intrinsic matrix.

For more detail, please take a look at [Deep Learning-based 6-DoF Grasp Estimation for Industrial Bin-Picking](https://drive.google.com/file/d/13bMaAVxF_TxXuYZXmq5LAVM3r0F_-B0Z/view?usp=sharing) (in the process of submission). 

<p align="center">
  <img width=100% height=100% src="https://github.com/AdamPurnomo/Deep-Learning-based-Grasp-Planning/blob/main/Images/overview.png?raw=true">
</p>

## Network Architecture
The network takes inputs of individual grasp image and outputs the grasping score and the 2D projection of the grasp approaching pose vector. The network is divided into three parts: feature extractor, grasp pose estimator and grasp quality estimator.

<p align="center">
  <img width=100% height=100% src="https://github.com/AdamPurnomo/Deep-Learning-based-Grasp-Planning/blob/main/Images/CNNarch.png">
</p>

## Folder Navigation
Description of each folder

1. Development
This folder contains all files for the development of the system. It has 3 subfolders.
  * Network
    contains training data, test data, and script for training and test the network
  * Simulation
    contains script for performing simulation and generating synthetic data. The generated
    synthetic data is saved inside the Image folder. The main file for creating synthetic data is `synthetic_data_generation.py`.
    The rest of them are modules.
  * Utility
    cotains python module used for data preprocessing
All python script modules contain documentation of each function. Please take a look at the particular python file 
to look more in detail what each function does.
	
2. Execution
This folder contains all files required for performing experiments. The controller software is developed by our lab. Please take a look at `inference.py` for the inference part. All python script modules contain documentation of each function. Please take a look at the particular python file to look more in detail what each function does.

## Experiments

This method is validated with 4 types of industrial objects. The full video experiment can be seen [here](https://youtu.be/J0o0fcqUbLQ).

### Object A
<p align="center">
  <img width=100% height=100% src="https://github.com/AdamPurnomo/Deep-Learning-based-Grasp-Planning/blob/main/Images/object%20A%20success.png">
</p>

### Object B
<p align="center">
  <img width=100% height=100% src="https://github.com/AdamPurnomo/Deep-Learning-based-Grasp-Planning/blob/main/Images/object%20B%20success.png">
</p>

### Object C
<p align="center">
  <img width=100% height=100% src="https://github.com/AdamPurnomo/Deep-Learning-based-Grasp-Planning/blob/main/Images/object%20C%20success.png">
</p>

### Object D
<p align="center">
  <img width=100% height=100% src="https://github.com/AdamPurnomo/Deep-Learning-based-Grasp-Planning/blob/main/Images/object%20D%20success.png">
</p>

## Dependencies
* tensor flow 2.5.0
* numpy 1.19.4
* opencv-python 4.4.0.46
* pillow 8.0.1
* trimesh 3.9.19
* pybullet 3.0.7

