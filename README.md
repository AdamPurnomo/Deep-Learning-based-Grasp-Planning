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
  <img width=100% height=100% src="![image](https://github.com/AdamPurnomo/Deep-Learning-based-Grasp-Planning/blob/main/Images/CNNarch.png)">
</p>
