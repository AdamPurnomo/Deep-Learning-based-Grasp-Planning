# %%
import pybullet as p
from scipy.stats import mode
import numpy as np



def clipspaceposition (pos,  depthmap, parameters):
  '''
  Transform position in pixel space to clip space.
  # input 
  pos         : position in pixel space. 
                Shape (n,2)
  depthmap    : depth map
                Shape (m,n)
  parameters  : Open gl rendering parameters as follow.
                 - Far range / furthest camera reach
                 - near range / closest camera reach
  # output
  clippos     : position in clip space.
                Shape (n,4)
  '''
  far = parameters[0]
  near = parameters[1]

  i = pos[:,0]
  j = pos[:,1]
  row_size, col_size = depthmap.shape

  x_n = (2/(col_size - 1))*i - 1
  y_n = 1 - (2/(row_size - 1))*j
  buffer_value = depthmap[j,i]

  x_c = x_n*buffer_value
  y_c = y_n*buffer_value
  z_c = ((far+near)*buffer_value - 2*far*near) / (far-near)
  w_c = buffer_value

  clippos = np.array([x_c, y_c, z_c, w_c]).transpose()

  return clippos


def realposition (clippos, viewMatrix, projectionMatrix):
  '''
  Transform position in clip space to 3d cartesian coordinate .
  # input 
  clippos           : position in clip space. 
                      Shape (n,4)
  viewMatrix        : Open gl view matrix
                      Shape (4,4)
  projectionMatrix  : Open gl projection matrix
                      Shape (4,4)
  # output
  realpos     : position in 3d cartesian coordinate.
                Shape (n,3)
  '''
  P_inv = np.linalg.inv(projectionMatrix)
  V_inv = np.linalg.inv(viewMatrix)

  viewpos = P_inv.dot(clippos.T)
  realpos = V_inv.dot(viewpos)
  realpos = realpos[0:3,:] / realpos[3,:]

  return realpos

def to_depthMap (depth_buffer, parameters):
  '''
  Convert opengl depth buffer into depth map
  #input
  depth_buffer : depth buffer from open gl rendering.
                 Shape (m,n)
  parameters   : Open gl rendering parameters as follow.
                 - Far range / furthest camera reach
                 - near range / closest camera reach
  #output
  depth_map    : depth map of a scene in a simulated env
                 Shape (m,n)
  '''
  far = parameters[0]
  near = parameters[1]
  depth_map = far * near / (far - (far - near) * depth_buffer)
  return depth_map

def grasps_projection(grasps, depthmap, parameters, rendering_matrices):
  '''
  Convert grasps coordinate from pixel coordinate to 3d cartesian coordinate.
  #input
  grasps            : grasp candidates in pixel coordinate
                      Shape (n, 2, 2)
  depth_map         : depth map
                      Shape (m,n)
  parameters        : Open gl rendering parameters as follow.
                      - Far range / furthest camera reach
                      - near range / closest camera reach
  rendering_matrices : set of matrices to render a scene into an image 
                       which consist of 
                        - view matrix
                        - projection matrix
  #output
  projected_grasps    : depth map of a scene in a simulated env
                        Shape (n,2,3)
  '''
  
  viewMatrix = rendering_matrices[0]
  projectionMatrix = rendering_matrices[1]
  p1 = grasps[:,0]
  p2 = grasps[:,1]

  clip_p1 = clipspaceposition(p1, depthmap,parameters)
  clip_p2 = clipspaceposition(p2, depthmap,parameters)

  real_p1 = np.expand_dims(realposition(clip_p1, viewMatrix, projectionMatrix).T,axis = 1)
  real_p2 = np.expand_dims(realposition(clip_p2, viewMatrix, projectionMatrix).T, axis = 1)
  projected_grasps = np.concatenate((real_p1, real_p2), axis = 1)
  return projected_grasps
  
def orientation_correction(p_grasps, centers, segImg, partID, position, orientation):
  '''
  Since we lose some spatial information in 2D image, the projection from
  2D image to 3D cartesian coordinate will not likely be accurate. Therefore,
  further correction is needed according to the orientation of the object.

  #input
  p_grasps    : Set of grasp candidates in 3D cartesian coordinate
                Shape (n, 2, 3)
  centers     : Set of pixel location between two contact points in each grasp candidates
                Shape (n, 4, 2)
  segImg      : Segmentation image
                Shape (m,n)
  partID      : Set of object ID inside the bin
                Shape (num. of object, )
  position    : position of each object in 3D cartesion coordinate
                Shape (num. of object, 3)
  orientation : Rotation matrix of each object in the bin
                Shape (num, of object, 3, 3)
  
  #Output
  c_grasps    : Set of corrected grasp candidates position in 3D cartesian coordinate
                Shape (n, 2, 3)
  T_set       : Set of transformation matrix of each object inside the bin
                Shape (num. of object, 4, 4)
  grasp_ID    : Set of associated object ID of each grasp candidates 


  '''
  #Associating each grasp candidates with each object ID in the bin
  graspID = mode(segImg[centers[:,:,0], centers[:,:,1]], axis= 1)[0]
  T_set = []
  for i, ID in enumerate(partID):
    ob_pos = position[i]
    ob_rot_matrix = orientation[i]

    #Object Local coordinate
    z = np.array([0,0,1])
    y = np.array([0,1,0])
    x = np.array([1,0,0])

    #Transforming object local coordinate to universal coordinate
    ob_cor = np.array([np.dot(ob_rot_matrix, x) , 
                      np.dot(ob_rot_matrix, y) , 
                      np.dot(ob_rot_matrix, z)])
    
    #Finding out which object's axes is closer to universal z-axis
    dt_pr = np.array([abs(np.dot(ob_cor[0], z)) , abs(np.dot(ob_cor[1], z)) , abs(np.dot(ob_cor[2], z))])
    index = np.argmax(dt_pr) #index of plane normal vector

    dindex, _ = np.where(graspID == ID)

    z0 = (((ob_pos[0]-p_grasps[dindex,0,0])*ob_cor[index,0] + 
           (ob_pos[1]-p_grasps[dindex,0,1])*ob_cor[index,1] + 
            ob_pos[2]*ob_cor[index][2]) / ob_cor[index,2])
    
    z1 = (((ob_pos[0]-p_grasps[dindex,1,0])*ob_cor[index,0] + 
           (ob_pos[1]-p_grasps[dindex,1,1])*ob_cor[index,1] + 
            ob_pos[2]*ob_cor[index][2]) / ob_cor[index,2])

    
    p_grasps[dindex,0,2] = z0
    p_grasps[dindex,1,2] = z1

    #Generating transformation matrix for each object in the bin
    T_matrix = np.zeros((4,4))
    T_matrix[3,3] = 1
    T_matrix[0:3,0:3] = ob_rot_matrix
    T_matrix[0:3,3] = ob_pos
        
    T_set.append(T_matrix)
   
  return p_grasps, T_set, graspID

    
def grasp_local_transformation(T_set, u_grasps, graspID_index):
  '''
  Transformation of grasp candidates from universal coordinate
  to object local coordinate
  #input
  T_set   : Set of transformation matrices
            Shape (num. of object, 4, 4)
  u_grasps  : grasps position in universal coordinate
              Shape (n, 2, 3)
  graspID_index : indexes of graspID in part ID
                  Shape (n, 1)
  
  #output
  l_grasps : Set of grasp candidates in object local coordinate
             Shape (n, 2, 3)
  '''
  
  l_grasps = np.zeros(u_grasps.shape)
  for i, T_matrix in enumerate(T_set):
    T_inv = np.linalg.inv(T_matrix)
    index, _ = np.where(graspID_index == i)

    p1 = np.ones(4, len(u_grasps[index]))
    p2 = np.ones(4, len(u_grasps[index]))
    p1[0:3, :] = u_grasps[index, 0, :].T
    p2[0:3, :] = u_grasps[index, 1, :].T
    p1 = 0.1*T_inv.dot(p1)
    p2 = 0.1*T_inv.dot(p2)

    l_grasps[index, 0, :] = p1[0:3,:].T
    l_grasps[index, 1, :] = p2[0:3,:].T

  return l_grasps


def grasp_univ_transformation(T_set, l_grasps, graspID_index):
  '''
  Transformation of grasp candidates from object local coordinate
  to universal coordinate
  #input
  T_set   : Set of transformation matrices
            Shape (num. of object, 4, 4)
  l_grasps  : grasps position in object local coordinate
              Shape (n, 2, 3)
  graspID_index : indexes of graspID in part ID
                  Shape (n, 1)
  
  #output
  u_grasps : Set of grasp candidates in object universal coordinate
             Shape (n, 2, 3)
  '''
  
  u_grasps = np.zeros(l_grasps.shape)
  for i, T_matrix in enumerate(T_set):
    index, _ = np.where(graspID_index == i)

    p1 = np.ones(4, len(l_grasps[index]))
    p2 = np.ones(4, len(l_grasps[index]))
    p1[0:3, :] = l_grasps[index, 0, :].T
    p2[0:3, :] = l_grasps[index, 1, :].T
    p1 = 10*T_matrix.dot(p1)
    p2 = 10*T_matrix.dot(p2)

    u_grasps[index, 0, :] = p1[0:3,:].T
    u_grasps[index, 1, :] = p2[0:3,:].T

  return u_grasps
   

def draw_realgrasps(ux_grasps):
   for i in range(len(ux_grasps)):
     center = (ux_grasps[i,0] + ux_grasps[i,1]) / 2
     ux_grasps[i,0] = center + 2*(ux_grasps[i,0] - center)
     ux_grasps[i,1] = center + 2*(ux_grasps[i,1] - center)
     
     start = ux_grasps[i,1]
     direction = ux_grasps[i,0] - ux_grasps[i,1]
     finish = start + 1*direction
     p.addUserDebugLine(start, finish, [1,0,0], lineWidth = 4.0)
    
  
"""
### Debugging
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane100.urdf")
cube = p.loadURDF("r2d2.urdf", [-2.8, 2.8, 0])



  
fov = 45.0
aspect = 1
near = 0.1
far = 7.1
viewMatrix = p.computeViewMatrix(
    cameraEyePosition=[0,0,7],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 1, 0])

projectionMatrix = p.computeProjectionMatrixFOV(
    fov=fov,
    aspect=1.0,
    nearVal=near,
    farVal=far)

width, height, rgbImg, depthImg, segImg = p.getCameraImage(
    width=224, 
    height=224,
    viewMatrix=viewMatrix,
    projectionMatrix=projectionMatrix,
    renderer=p.ER_BULLET_HARDWARE_OPENGL)

#V = np.transpose(np.reshape(viewMatrix,(4,4)))
#V_inv = np.linalg.inv(V)
#print(V_inv)
depth_buffer_opengl = np.reshape(depthImg, [224, 224])
depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)

clippos_center = clipspaceposition(0, 0, 224, 224, depth_opengl, far, near)
realpos_ = realposition(clippos_center, viewMatrix, projectionMatrix)
"""









# %%
