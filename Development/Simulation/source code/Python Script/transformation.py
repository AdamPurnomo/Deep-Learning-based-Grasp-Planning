# %%
import numpy as np
import pybullet as p


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
  
def orientation_correction(p_grasps, segImg, position, orientation):
  '''
  Since we lose some spatial information in 2D image, the projection from
  2D image to 3D cartesian coordinate will not likely be accurate. Therefore,
  further correction is needed according to the orientation of the object.

  #input
  p_grasps    : Set of grasp candidates in 3D cartesian coordinate
                Shape (n, 2, 3)
  segImg      : Segmentation image
                Shape (m,n)
  position    : position of each object in 3D cartesion coordinate
                Shape (3, )
  orientation : Rotation matrix of each object in the bin
                Shape (3, 3)
  
  #Output
  c_grasps    : Set of corrected grasp candidates position in 3D cartesian coordinate
                Shape (n, 2, 3)
  T_matrix       : Transformation matrix 
                Shape (num. of object, 4, 4)
  '''

  ob_pos = position
  ob_rot_matrix = orientation

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


  z0 = (((ob_pos[0]-p_grasps[:,0,0])*ob_cor[index,0] + 
           (ob_pos[1]-p_grasps[:,0,1])*ob_cor[index,1] + 
            ob_pos[2]*ob_cor[index][2]) / ob_cor[index,2])
    
  z1 = (((ob_pos[0]-p_grasps[:,1,0])*ob_cor[index,0] + 
           (ob_pos[1]-p_grasps[:,1,1])*ob_cor[index,1] + 
            ob_pos[2]*ob_cor[index][2]) / ob_cor[index,2])

    
  p_grasps[:,0,2] = z0
  p_grasps[:,1,2] = z1

  #Generating transformation matrix 
  T_matrix = np.zeros((4,4))
  T_matrix[3,3] = 1
  T_matrix[0:3,0:3] = ob_rot_matrix
  T_matrix[0:3,3] = ob_pos
        
  
   
  return p_grasps, T_matrix

    
def grasp_local_transformation(T_matrix, u_grasps):
  '''
  Transformation of grasp candidates from universal coordinate
  to object local coordinate
  #input
  T_set   : Transformation matrix
            Shape (num. of object, 4, 4)
  u_grasps  : grasps position in universal coordinate
              Shape (n, 2, 3)

  #output
  l_grasps : Set of grasp candidates in object local coordinate
             Shape (n, 2, 3)
  '''
  
  l_grasps = np.zeros(u_grasps.shape)
  T_inv = np.linalg.inv(T_matrix)

  p1 = np.ones((4, len(u_grasps)))
  p2 = np.ones((4, len(u_grasps)))
  p1[0:3, :] = u_grasps[:, 0, :].T
  p2[0:3, :] = u_grasps[:, 1, :].T
  p1 = T_inv.dot(p1)
  p2 = T_inv.dot(p2)

  l_grasps[:, 0, :] = p1[0:3,:].T
  l_grasps[:, 1, :] = p2[0:3,:].T

  return l_grasps


def grasp_univ_transformation(T_matrix, l_grasps):
  '''
  Transformation of grasp candidates from object local coordinate
  to universal coordinate
  #input
  T_set   : Set of transformation matrices
            Shape (num. of object, 4, 4)
  l_grasps  : grasps position in object local coordinate
              Shape (n, 2, 3)

  #output
  u_grasps : Set of grasp candidates in object universal coordinate
             Shape (n, 2, 3)
  '''
  
  u_grasps = np.zeros(l_grasps.shape)
  p1 = np.ones((4, len(l_grasps)))
  p2 = np.ones((4, len(l_grasps)))
  p1[0:3, :] = l_grasps[:, 0, :].T
  p2[0:3, :] = l_grasps[:, 1, :].T
  p1 = T_matrix.dot(p1)
  p2 = T_matrix.dot(p2)

  u_grasps[:, 0, :] = p1[0:3,:].T
  u_grasps[:, 1, :] = p2[0:3,:].T

  return u_grasps

    
def to_pixel(grasps, rendering_matrices, img_size):
    '''
    Transform grasp candidates in 3d cartesian coordinate to pixel space
    #input
    grasps              : grasp candidates in 3d cartesian coordinate
                          Shape (n, 2, 3)
    rendering_matrices  : set of rendering matrices as follows
                            - view matrix (4,4)
                            - projection matrix (4,4)
    img_size            : size of image of the scene
                         Shape (2,)

    #output
    t_grasps            : projected grasps in image space
                          Shape (n, 2, 2)
    '''
    view_matrix = rendering_matrices[0]
    projection_matrix = rendering_matrices[1]
    row_size, column_size = img_size
    v1 = np.ones((4, len(grasps)))
    v2 = np.ones((4, len(grasps)))
    v1[0:3,:] = grasps[:,0,:].T
    v2[0:3,:] = grasps[:,1,:].T
    viewpos1 = view_matrix.dot(v1)
    viewpos2 = view_matrix.dot(v2)
    clippos1 = projection_matrix.dot(viewpos1)
    clippos2 = projection_matrix.dot(viewpos2)
    
    normalize1 = clippos1[0:2,:] / clippos1[3,:] #(3,m)
    normalize2 = clippos2[0:2,:] / clippos2[3,:]

    #remove grasping candidates outside the scene
    clipped_indices = np.where((normalize1>1)  |
                               (normalize1<-1) |
                               (normalize2>1)  |
                               (normalize2<-1))[1]
    

    column1 = (normalize1[0,:] + 1)*(column_size-1)/2 #column index of contact point 1 in the image pixel
    column2 = (normalize2[0,:] + 1)*(column_size-1)/2 #column index of contact point 2 in the image pixel
    row1 = (1 - normalize1[1,:])*(row_size-1)/2 #row index of contact point 1 in the image pixel
    row2 = (1 - normalize2[1,:])*(row_size-1)/2 #row index of contact point 2 in the image pixel

    px1 = np.array([column1, row1]) #(2,m)
    px2 = np.array([column2, row2])
    t_grasps = np.hstack((px1.T, px2.T)).reshape(-1,2,2) #reshaped from (m,4) the grasp is represented in ((column, row), (column, row))
    t_grasps = np.delete(t_grasps, clipped_indices, axis = 0)
    return t_grasps, clipped_indices
   

def draw_realgrasps(ux_grasps, grasp_scores):
   for i in range(len(ux_grasps)):
     center = (ux_grasps[i,0] + ux_grasps[i,1]) / 2
     start = center + 2*(ux_grasps[i,0] - center)
     end  = center + 2*(ux_grasps[i,1] - center)
     
     direction = end - start
     finish = start + direction
     p.addUserDebugLine(start, finish, [0,0,1], lineWidth = 4.0)
     #p.addUserDebugLine(start, finish, [1*(1-grasp_scores[i]),1*grasp_scores[i],0], lineWidth = 4.0)
    
def to_pointmap(depthmap, far, near, P, V):
  '''
  Generate point map from camera scene
  #input
  depthmap  : depth map
             shape(m,n)
  far       : far perspective point
  near      : near perspective point
  P         : projection matrix
  V         : view matrix

  #Output
  xr, yr, zr  : point map of x, y, z from the camera scene
  '''
  row, column = depthmap.shape
  xp = np.arange(0,column,1)
  yp = np.arange(0,row,1)
  xp,yp = np.meshgrid(xp,yp)
  xp = xp.flatten()
  yp = yp.flatten()
  px = np.array([xp,yp]).T 
  cp = clipspaceposition(px, depthmap,[far,near])
  real = realposition(cp, V, P)
  xr = real[0,:]
  yr = real[1,:]
  zr = real[2,:]
  xr = xr.reshape((row, column))
  yr = yr.reshape((row, column))
  zr = zr.reshape((row, column))

  return xr, yr, zr







# %%
