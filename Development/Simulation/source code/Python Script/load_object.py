



import numpy as np
import pybullet as pb
import trimesh as tr

#constant variables
container_loc = r'..\..\..\3D object data\box_v1.obj' 
pi = np.pi/180

def load_parts (position, orientation, part_dir, scale):
    '''
    Load the target object into the simulation environment.
    #input
    position    : position of the object in the simulation environment
                  Shape (3,)
    orientation : Euler angels of the object in the simulation environment
                  Shape (3,)
    part_dir    : Relative directory of the object in the folder
                 [.OBJ directory, .STL directory]
    scale       : scalar value that determines the scale of the object in pybullet env

    #output 
    part        : ID associated with the part
    '''
    part_loc = part_dir[0]
    stl_loc = part_dir[1]
    mesh = tr.load_mesh(stl_loc, process = False)
    CoMshift = mesh.center_mass
    
    visshape = pb.createVisualShape(shapeType = pb.GEOM_MESH,
                                fileName  = part_loc,
                                meshScale = [1*scale, 1*scale, 1*scale]
                                )

    colshape = pb.createCollisionShape(shapeType = pb.GEOM_MESH,
                             fileName  = part_loc,
                             meshScale = [1*scale, 1*scale, 1*scale]
                             )

    part = pb.createMultiBody(
        baseMass = 1,
        baseInertialFramePosition = scale*CoMshift,
        baseCollisionShapeIndex = colshape,
        baseVisualShapeIndex = visshape,
        basePosition = position,
        baseOrientation = pb.getQuaternionFromEuler(orientation)
    )

    return part    

def load_container ():
    '''
    Load the container into the simulation environment.
    
    #input
    None

    #output 
    cont     : ID of the container
    '''
    visshape = pb.createVisualShape(shapeType = pb.GEOM_MESH,
                                fileName  = container_loc,
                                meshScale = [0.015, 0.015, 0.01]
                                )
    colshape = pb.createCollisionShape(shapeType = pb.GEOM_MESH,
                             fileName  = container_loc,
                             meshScale = [0.015, 0.015, 0.01],
                                flags = pb.GEOM_FORCE_CONCAVE_TRIMESH
                             )

    cont = pb.createMultiBody(
        baseMass = 20,
        baseCollisionShapeIndex = colshape,
        baseVisualShapeIndex = visshape,
        basePosition = (0,0,0),
        baseOrientation = pb.getQuaternionFromEuler((0,0,0))
        )
    return cont
    
def drawAABB(aabb):
    '''
    Draw AABB bounding box of the object inside the simulation environment
    #input
    aabb    : axis alligned bounding box
            [Minimum box, maximum box]
    
    #output
    None
    '''
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

