# %%
import trimesh as tr 
import numpy as np 
import pybullet as pb 
import load_object as lo
import pybullet_data
import antipodal_sampler as ap 
import wrench_space_analysis as wsa 
import transformation as t 
import random as r 
import metrics as m 
import json

#need debugging with the scale and scaleinv

with open(r'..\JSON File\parameters.json') as f:
    parameters = json.load(f)

obj = r'..\..\..\3D object data\cylinder.obj'
arrow = r'..\..\..\3D object data\arrow.obj'
gripper = r'..\..\..\3D object data\Gripper.obj'

def load_parts (position, orientation, part_dir, scale):
    '''
    load object to simulation env. The difference between this function and load_parts function
    in load object module is that, this function is specialized to load parts with more accurate 
    collision shape. The drawback with more accurate collision shape is that it can only be applied
    for static object. 

    #input
    position    : Position of the object in 3d cartesian coordinate
                  Shape (3, )
    orientation : orientation of the object represented in euler angle
                  Shape (3, )
    part_dir    : relative path of object file directory. Contains two paths,
                  the first one is for .obj file and the second one is for .stl file
    scale       : scalar value that determines the scale of the object in pybullet env

    #output
    part        : ID associated with the object
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
                             meshScale = [1*scale, 1*scale, 1*scale],
                             flags = pb.GEOM_FORCE_CONCAVE_TRIMESH
                             )

    part = pb.createMultiBody(
        baseMass = 0,
        baseInertialFramePosition = scale*CoMshift,
        baseCollisionShapeIndex = colshape,
        baseVisualShapeIndex = visshape,
        basePosition = position,
        baseOrientation = pb.getQuaternionFromEuler(orientation)
    )
    return part

def attach_cylinder(quaternion, position):
    '''
    Loading cylinder amd attach it to the grasp contact 
    as representation of the end effector stroke. 
    The cylinder will be used to figure out whether the grasping stroke
    will collide with the object or not.

    #input 
    quaternion  : cylinder orientation represented in quaternion
                  Shape (4, )
    position    : the position where the cylinder face will be attached
    
    #output
    ID          : id associated with cylinder
    '''
    visshape = pb.createVisualShape(shapeType = pb.GEOM_MESH,
                                fileName  = obj,
                                meshScale = [0.0016, 0.001, 0.0016]
                                )

    colshape = pb.createCollisionShape(shapeType = pb.GEOM_MESH,
                             fileName  = obj,
                             meshScale = [0.0016, 0.001, 0.0016]
                             )

    ID = pb.createMultiBody(
        baseMass = 1,
        baseCollisionShapeIndex = colshape,
        baseVisualShapeIndex = visshape,
        basePosition = position,
        baseOrientation = quaternion)
    return ID
    

def calculate_quaternion(u, v):
    '''
    Calculate quaternion to rotate one vector to
    the direction of another vector

    #input
    u       : vector to be rotated (3, )
    v       : direction of where the vector rotated to (3, )
    
    #output
    quaternion  : quaternion to rotated the vector (4, )
    '''
    rotaxis = np.cross(u,v)
    rotaxis = rotaxis / np.linalg.norm(rotaxis)

    rotangle = np.arccos(u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    quaternion = np.array([rotaxis[0] * np.sin(rotangle/2),
                       rotaxis[1] * np.sin(rotangle/2),
                       rotaxis[2] * np.sin(rotangle/2),
                       np.cos(rotangle/2)])
    return quaternion

def collision_detection(grasps, ObjID, visualize):
    '''
    Detect whether the grasping stroke collide with the object or not.
    
    #input 
    grasps  : grasp candidates
              Shape (n, 2, 3)
    ObjID   : id associated with the object to be grasped
    visualize : bool data to decide whether to visualize the stroke or not

    #output
    scores  : binary scores for each grasp candidates. 
              1 means the grasp candidates will collide
              and 0 means the stroke is free collision
              Shape (n, ) 

    '''
    scores = []
    for i in range(len(grasps)):
        u = np.array([0, 1, 0])
        v1 = grasps[i,1] - grasps[i,0]
        v0 = -v1

        quaternion1 = calculate_quaternion(u ,v1)
        quaternion0 = calculate_quaternion(u, v0)
        c1 = attach_cylinder(quaternion1, grasps[i,1] + 0.02*v1)
        c0 = attach_cylinder(quaternion0, grasps[i,0] + 0.02*v0)

        collision_0 = len(pb.getClosestPoints(c0, ObjID, 0.001))
        collision_1 = len(pb.getClosestPoints(c1, ObjID, 0.001))

        if(collision_0 > 0 or collision_1 > 0):
            score = 1
        else:
            score = 0

        if(visualize == True):
            pb.changeVisualShape(c0, -1, rgbaColor=[score*1, (1-score)*1, 0, (1-score)*1])
            pb.changeVisualShape(c1, -1, rgbaColor=[score*1, (1-score)*1, 0, (1-score)*1])
        scores.append(score)
    return np.asarray(scores)




def sample_valid_grasps(part_name, visualize,scaleinv):
    '''
    Sample grasp candidates which its stroke does not collide
    with the target object

    #input 
    part_name   : name of the target object
                  type str
    visualize   : bool data to visualize the grasping stroke
                  type bool
    scale       : inverse scale differencet from stl to obj
    #output
    grasps      : a set of valid grasp candidates
    normlas     : normal vectors associated with each grasp
    '''
    #initializing parameters
    part_parameters = parameters[part_name]
    ##volumegrasp_range = [scaleinv*part_parameters[2], scaleinv*part_parameters[3]] 
    volumegrasp_range = [part_parameters[2], part_parameters[3]] 
    part_dir = [part_parameters[7], part_parameters[8]]


    #opening simulation client
    if(visualize == True):
        physicsClient = pb.connect(pb.GUI)
    else:
        physicsClient = pb.connect(pb.DIRECT)

    pb.setAdditionalSearchPath(pybullet_data.getDataPath()) 
    
    #calculate center of mass
    mesh = tr.load_mesh(part_dir[1], process = False)
    CoMshift = mesh.center_mass
    bb_extent = mesh.bounding_box.primitive.extents

    #loading part into env
    ObjID = load_parts(-(10/scaleinv)*CoMshift, [0,0,0], part_dir,10/scaleinv)

    #3d grasps sampling
    grasps, normals, face_indices = ap.volume_antipodal_sampler(part_dir[1], volumegrasp_range)
    indices = np.arange(0, len(grasps))
    r.shuffle(indices)
    ##grasps = (10/scaleinv)*grasps
    grasps = (10)*grasps

    #pruning grasps stroke that will collide with the object
    scores = collision_detection(grasps, ObjID, visualize)

    validindices = np.where(scores == 0)
    grasps = grasps[validindices]
    normals = normals[validindices]

    #calculate wrench space metric score and prune invalid ones
    t_scal = wsa.torque_scale(bb_extent)
    grasp_scores, l_scores, invalid_id = wsa.evaluate_grasps(grasps, normals, CoMshift, t_scal)
    np.delete(grasps, invalid_id, axis=0)
    np.delete(normals,invalid_id, axis=0)

    #saving grasp candidates and its scores
    data = np.array([grasps, normals])
    np.save(r'../Npy File/local grasps/'+part_name+r'/local_grasps.npy',data)
    scores = np.array([grasp_scores, l_scores])
    np.save(r'../Npy File/local grasps/'+part_name+r'/scores.npy',scores)
    
    pb.disconnect()
    return grasps, normals


def load_gripper(orientation, position):
    '''
    Loading cylinder amd attach it to the grasp contact 
    as representation of the end effector stroke. 
    The cylinder will be used to figure out whether the grasping stroke
    will collide with the object or not.

    #input 
    quaternion  : cylinder orientation represented in quaternion
                  Shape (4, )
    position    : the position where the cylinder face will be attached
    
    #output
    ID          : id associated with cylinder
    '''
    offset = np.array([-0.5,0, 0.1])
    orientation_offset = np.array([0,90,90])*np.pi/180
    R = Z_R(orientation_offset[2]).dot(Y_R(orientation_offset[1]))
    offset = R.dot(offset)
    position = position + offset 
    orientation =  orientation_offset
    visshape = pb.createVisualShape(shapeType = pb.GEOM_MESH,
                                fileName  = gripper,
                                meshScale = [0.004, 0.004, 0.004]
                                )

    colshape = pb.createCollisionShape(shapeType = pb.GEOM_MESH,
                             fileName  = gripper,
                             meshScale = [0.004, 0.004, 0.004]
                             )

    ID = pb.createMultiBody(
        baseMass = 1,
        baseCollisionShapeIndex = colshape,
        baseVisualShapeIndex = visshape,
        basePosition = position,
        baseOrientation =  pb.getQuaternionFromEuler(orientation))
    return ID


def Z_R(psi):
    '''
    Rotation matrix around Z-axis 
    
    #input
    psi     : rotation angle
    #output
    Z       : rotation matrix around z axis
    '''
    Z = np.eye(3)
    Z[0,0] = np.cos(psi)
    Z[0,1] = -np.sin(psi)
    Z[1,0] = np.sin(psi)
    Z[1,1] = np.cos(psi)
    return Z

def Y_R(theta):
    '''
    Rotation matrix around Y-axis 
    
    #input
    theta     : rotation angle
    #output
    Y       : rotation matrix around y axis
    '''
    Y = np.eye(3)
    Y[0,0] = np.cos(theta)
    Y[0,2] = np.sin(theta)
    Y[2,0] = -np.sin(theta)
    Y[2,2] = np.cos(theta)
    return Y

def X_R(phi):
    '''
    Rotation matrix around X-axis 
    
    #input
    phi     : rotation angle
    #output
    X       : rotation matrix around x axis
    '''
    X = np.eye(3)
    X[1,1] = np.cos(phi)
    X[1,2] = -np.sin(phi)
    X[2,1] = np.sin(phi)
    X[2,2] = np.cos(phi)
    return X




# %%
