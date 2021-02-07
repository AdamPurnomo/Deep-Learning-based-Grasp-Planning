



#import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb
import trimesh as tr



container_loc = r'..\..\..\data\model\Container_v1.obj' 
pi = np.pi/180


def load_parts (position, orientation, part_dir):
    part_loc = part_dir[0]
    stl_loc = part_dir[1]
    mesh = tr.load_mesh(stl_loc, process = False)
    CoMshift = 10*mesh.center_mass
    
    visshape = pb.createVisualShape(shapeType = pb.GEOM_MESH,
                                fileName  = part_loc,
                                meshScale = [10, 10, 10]
                                )

    colshape = pb.createCollisionShape(shapeType = pb.GEOM_MESH,
                             fileName  = part_loc,
                             meshScale = [10, 10, 10],
                             flags = pb.GEOM_FORCE_CONCAVE_TRIMESH
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
                                meshScale = [0.004, 0.008, 0.006]
                                )
    colshape = pb.createCollisionShape(shapeType = pb.GEOM_MESH,
                             fileName  = container_loc,
                             meshScale = [0.004, 0.008, 0.006],
                                flags = pb.GEOM_FORCE_CONCAVE_TRIMESH
                             )

    cont = pb.createMultiBody(
        baseMass = 20,
        baseCollisionShapeIndex = colshape,
        baseVisualShapeIndex = visshape,
        basePosition = (-2,-2,0),
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

r""" 
    ### Generate Depth Image
physicsClient = pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = pb.loadURDF('plane.urdf')
pb.setGravity(0,0,-98)

container = load_container()

partID = []

for i in range (1):
    position = [0, 0, i+1]
    orientation = [r.uniform(-pi*360,pi*360), r.uniform(-pi*360,pi*360), r.uniform(-pi*360,pi*360)]
    partID.append(load_parts(position, orientation))
    


for i in range (1000):
    pb.stepSimulation(physicsClient)
    time.sleep(1./240.)

viewMatrix = pb.computeViewMatrix(
    cameraEyePosition=[0, 0, 6],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 1, 0])

projectionMatrix = pb.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=8.1)

width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
    width=224, 
    height=224,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix)

plt.imsave(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\depth.png', depthImg[67:155, 13:211], cmap='gray_r')
plt.imsave(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\Segmentation.png', segImg[67:155, 13:211])

pb.disconnect()

"""

r"""
### Generate Depth Image
physicsClient = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = pb.loadURDF('plane.urdf')
pb.setGravity(0,0,-98)

container = load_container()

partID = []

for i in range (30):
    position = [0, 0, i+1]
    orientation = [r.uniform(-pi*360,pi*360), r.uniform(-pi*360,pi*360), r.uniform(-pi*360,pi*360)]
    partID.append(load_parts(position, orientation))
    


for i in range (1000):
    pb.stepSimulation(physicsClient)
    time.sleep(1./240.)

viewMatrix = pb.computeViewMatrix(
    cameraEyePosition=[0, 0, 6],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 1, 0])

projectionMatrix = pb.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=8.1)

width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
    width=224, 
    height=224,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix)

pb.disconnect()

#np.savetxt(r'depth.csv', depthImg, delimiter=',')

#plt.imsave(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\depth.png', depthImg[67:155, 13:211], cmap='gray_r')
#plt.imsave(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\Real.png', rgbImg)
#plt.imsave(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\Segmentation.png', segImg[67:155, 13:211])


with open('orientation.csv', mode='w') as orientation_file:
    orientation_writer = csv.writer(orientation_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for i in range (20):
        partpos, partorien = pb.getBasePositionAndOrientation(partID[i])
        orientation_writer.writerow([partorien[0], partorien[1], partorien[2]])

for i in range (20):
    aabb = pb.getAABB(partID[i])
    drawAABB(aabb)

        """
