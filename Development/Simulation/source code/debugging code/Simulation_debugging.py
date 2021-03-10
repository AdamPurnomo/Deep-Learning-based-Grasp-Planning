# %%

import matplotlib.pyplot as plt 
import numpy as np
import pybullet as pb
import pybullet_data
import trimesh as tr
import random as r
import time

part_loc = r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\data\model\hv18.obj'
container_loc = r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\data\model\Container_v1.obj' 
stl = r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\data\model\hv18.stl'
pi = np.pi/180
draw = 1
printtext = 0

mesh = tr.load_mesh(stl, process = False)
CoMshift = 10*mesh.center_mass 
size = 10*mesh.bounding_box.primitive.extents 

"""
The size in trimesh is defined as meter, while in pybullet, the size is in dm
"""

def load_parts (position, orientation):
    visshape = pb.createVisualShape(shapeType = pb.GEOM_MESH,
                                    fileName  = part_loc
                                    )

    colshape = pb.createCollisionShape(shapeType = pb.GEOM_MESH,
                                       fileName  = part_loc
                                       )

    part = pb.createMultiBody(
        baseMass = 1,
        baseInertialFramePosition = CoMshift,
        baseCollisionShapeIndex = colshape,
        baseVisualShapeIndex = visshape,
        basePosition = position,
        baseOrientation = pb.getQuaternionFromEuler(orientation)
    )

    return part
    
def load_container ():
    visshape = pb.createVisualShape(shapeType = pb.GEOM_MESH,
                                fileName  = container_loc,
                                meshScale = [0.004, 0.004, 0.004]
                                )
    colshape = pb.createCollisionShape(shapeType = pb.GEOM_MESH,
                             fileName  = container_loc,
                             meshScale = [0.004, 0.004, 0.004],
                                flags = pb.GEOM_FORCE_CONCAVE_TRIMESH
                             )

    cont = pb.createMultiBody(
        baseMass = 200,
        baseCollisionShapeIndex = colshape,
        baseVisualShapeIndex = visshape,
        basePosition = (-2,-1,0),
        baseOrientation = pb.getQuaternionFromEuler((0,0,0))
        )
    return cont

def drawAABB(aabb):
    aabbMin = aabb[0]
    aabbMax = aabb[1]
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMin[2]]
    pb.addUserDebugLine(f, t, [1, 0, 0])
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    pb.addUserDebugLine(f, t, [0, 1, 0])
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMin[1], aabbMax[2]]
    pb.addUserDebugLine(f, t, [0, 0, 1])
    
    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    pb.addUserDebugLine(f, t, [1, 1, 1])
    
    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    pb.addUserDebugLine(f, t, [1, 1, 1])
    
    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    pb.addUserDebugLine(f, t, [1, 1, 1])
    
    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    pb.addUserDebugLine(f, t, [1, 1, 1])
    
    f = [aabbMax[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    pb.addUserDebugLine(f, t, [1, 1, 1])
    
    f = [aabbMin[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    pb.addUserDebugLine(f, t, [1, 1, 1])
    
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    pb.addUserDebugLine(f, t, [1.0, 0.5, 0.5])
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    pb.addUserDebugLine(f, t, [1, 1, 1])
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    pb.addUserDebugLine(f, t, [1, 1, 1])



physicsClient = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = pb.loadURDF('plane.urdf')
pb.setGravity(0,0,-9.8)

container = load_container()

position = [0,0,1]
orientation = [r.uniform(-pi*360,pi*360), r.uniform(-pi*360,pi*360), r.uniform(-pi*360,pi*360)]
partID = load_parts(position, orientation)

for i in range (1000):
    pb.stepSimulation(physicsClient)
    time.sleep(1./240.)

far = 6.1
aspect = 1
near = 0.1
fov = 45.0
img_size = 224
    

viewMatrix = pb.computeViewMatrix(
    cameraEyePosition=[0, 0, 6],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 1, 0])

projectionMatrix = pb.computeProjectionMatrixFOV(
    fov=fov,
    aspect=aspect,
    nearVal=near,
    farVal=far)

width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
    width=img_size, 
    height=img_size,
    viewMatrix= viewMatrix,
    projectionMatrix=projectionMatrix)

position, orientation = pb.getBasePositionAndOrientation(partID)


