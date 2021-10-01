# Deep Learning-based 6-DoF Grasp Estimation for Industrial Bin-Picking

## Overview
This project focus on the problem of the 6-DoF parallel-jaw grasp for bin-picking in an industrial setting. We explore a method to estimate 6-DoF
grasp from bin-picking scenes in an industrial setting using a convolutional neural network (CNN) without having to estimate the 6D pose of the target object. We gained inspiration from the idea that the robot does not need to know exactly the
precise 6D pose of the target object to get a reasonably good grasp. A rough estimate on how to approach the object is often more than sufficient to grasp the target object. We usewhat we call grasp approaching pose vector which determines from which direction the robot gripper should approach the target object in Cartesian space.

