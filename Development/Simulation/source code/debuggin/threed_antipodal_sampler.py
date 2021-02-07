# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:07:40 2020

@author: Adam Syammas Zaki P
"""
import numpy as np
import trimesh as tr
import scipy.spatial.distance as ssd
import pybullet as pb
import pybullet_data
import Simulation as s
import Wrench_Space_Analysis as wsa
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
parameters = {'hv18': [10, 13, 0.02, 0.025, 60, 24, 700,
                       'cylindrical',
                       r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\data\model\hv18_2.obj', 
                       r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\data\model\hv18.stl',
                       'C:\\Users\\KosugeLab\\Desktop\\Adam\\Grasp_Planning\\Simulation\\Image\\Training Data\\Classification\\hv18\\Positive\\ ',
                       'C:\\Users\\KosugeLab\\Desktop\\Adam\\Grasp_Planning\\Simulation\\Image\\Training Data\\Classification\\hv18\\Negative\\ '], 
              'hv8':[38, 44, 0.09, 0.11, 60, 54, 1332,
                     'cicular',
                     r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\data\model\hv8.obj', 
                     r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\data\model\hv8.stl',
                     'C:\\Users\\KosugeLab\\Desktop\\Adam\\Grasp_Planning\\Simulation\\Image\\Training Data\\Classification\\hv8\\Positive\\ ',
                     'C:\\Users\\KosugeLab\\Desktop\\Adam\\Grasp_Planning\\Simulation\\Image\\Training Data\\Classification\\hv8\\Negative\\ ']}

def sample_surface(mesh, count):
    """
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Parameters
    ---------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    """

    # len(mesh.faces) float, array of the areas
    # of each face of the mesh
    area = mesh.area_faces
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces))
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples, face_index
    
def antipodal_sampler(stl, grasp_range, geometry):
    mesh = tr.load_mesh(stl)
    mesh.remove_degenerate_faces
    mesh.vertices -= mesh.center_mass
    samples, indices= sample_surface(mesh, 1000)
    normals = mesh.face_normals[indices]
    distance = ssd.squareform(ssd.pdist(samples))
    normals_ip = normals.dot(normals.T)
    valid_pairs = np.where((normals_ip < -0.9) & 
                       (distance < grasp_range[1]) &
                       (distance > grasp_range[0]))
    valid_pairs = np.c_[valid_pairs[0], valid_pairs[1]]
    contact_points1 = samples[valid_pairs[:,0], :]
    contact_points2 = samples[valid_pairs[:,1], :]
    normals_points1 = normals[valid_pairs[:,0], :]
    normals_points2 = normals[valid_pairs[:,1], :]
    indices1 = indices[valid_pairs[:,0]]
    indices2 = indices[valid_pairs[:,1]]
    v = contact_points1 - contact_points2
    v_norm = np.linalg.norm(v, axis=1)
    v = v / np.tile(v_norm[:, np.newaxis], [1, 3])
    ip1 = np.sum(normals_points1 * v, axis=1)
    ip2 = np.sum(normals_points2 * (-v), axis=1)
    if(geometry == 'cylindrical'):
        ip_principalaxis = v.dot(np.array([0,1,0]))
        antipodal_indices = np.where((ip1>np.cos(np.arctan(0.5)))&(ip2>np.cos(np.arctan(0.5)))&(ip_principalaxis>np.cos(45*np.pi/180)))[0]
    else:
        antipodal_indices = np.where((ip1>np.cos(np.arctan(0.5)))&(ip2>np.cos(np.arctan(0.5))))[0]
    
    grasp_indices = np.random.choice(antipodal_indices, size = antipodal_indices.shape[0], replace = False)
    p1 = contact_points1[grasp_indices, :]
    p2 = contact_points2[grasp_indices, :]
    n1 = normals_points1[grasp_indices, :]
    n2 = normals_points2[grasp_indices, :]
    i1 = indices1[grasp_indices]
    i2 = indices2[grasp_indices]
    grasps = np.hstack((p1,p2)).reshape(-1,2,3)
    normals = np.hstack((n1,n2)).reshape(-1,2,3)
    face_indices = np.hstack((i1,i2)).reshape(-1,2,1)
    return grasps, normals, face_indices
    
                     
def grasp_univ_transformation(T_matrix, lx_grasps):
    c1_vec = np.ones((4,len(lx_grasps))) #(4,m)
    c2_vec = np.ones((4,len(lx_grasps))) #(4,m)
    
    c1_vec[0:3,:] = 10*lx_grasps[:,0,:].T #(3,m) manipulating the matrix elements
    c2_vec[0:3,:] = 10*lx_grasps[:,1,:].T #(3,m) manipulating the matrix elements
    
    ux_c1 = ((T_matrix.dot(c1_vec))[0:3,:]).T #(m,3) The grasps location in pybullet environment
    ux_c2 = ((T_matrix.dot(c2_vec))[0:3,:]).T #(m,3) The grasps location in pybullet environment
    u_grasps = np.hstack((ux_c1, ux_c2)).reshape(-1,2,3)
    return u_grasps

def draw_realgrasps(ux_grasps, grasp_scores):
   for i in range(len(ux_grasps)):
     center = (ux_grasps[i,0] + ux_grasps[i,1]) / 2
     ux_grasps[i,0] = center + 2*(ux_grasps[i,0] - center)
     ux_grasps[i,1] = center + 2*(ux_grasps[i,1] - center)
     
     start = ux_grasps[i,1]
     direction = ux_grasps[i,0] - ux_grasps[i,1]
     finish = start + direction
     pb.addUserDebugLine(start, finish, [1*(1-grasp_scores[i]),1*grasp_scores[i],0], lineWidth = 4.0)

def visualization(part):
    grasp_range = parameters[part][2:4]
    stl_dir = parameters[part][9]
    obj_dir = parameters[part][8]
    part_dir = [obj_dir,stl_dir]
    geometry = parameters[part][7]
    
    grasps, normals, face_indices = antipodal_sampler(stl_dir, grasp_range, geometry)
    #ray_visualize = tr.load_path(grasps)
    #scene = tr.Scene([ray_visualize])
    #scene.show('gl')
    
    com = np.array([0,0,0])
    grasp_scores, invalid_indexes, loc_scores = wsa.evaluate_grasps(grasps, normals, com)
    grasps = np.delete(grasps, invalid_indexes, axis = 0)
    new_scores = 0.7*grasp_scores + 0.3*loc_scores
    
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF('plane.urdf')
    
    position = [0,0,0]
    orientation = [0,0,0]
    part = s.load_parts(position, orientation, part_dir)
    position, orientation = pb.getBasePositionAndOrientation(part)
    
    far = 6.1
    aspect = 1
    near = 0.1
    fov = 45.0
    img_size = 224

    viewMatrix = pb.computeViewMatrix(cameraEyePosition=[0, 0, 6],
                                      cameraTargetPosition=[0, 0, 0],
                                      cameraUpVector=[0, 1, 0])

    projectionMatrix = pb.computeProjectionMatrixFOV(fov=fov,
                                                     aspect=aspect,
                                                     nearVal=near,
                                                     farVal=far)

    width, height, rgbImg, depthImg, segImg = pb.getCameraImage(width=img_size, 
                                                                height=img_size,
                                                                viewMatrix= viewMatrix,
                                                                projectionMatrix=projectionMatrix)
    
    #depth_loc = r'C:\Users\KosugeLab\Desktop\Adam\Grasp_Planning\Simulation\Image\depth.png'
    #plt.imsave(depth_loc, depthImg, cmap='gray_r')
    
    viewMatrix = np.reshape(viewMatrix, (4,4), 'F')
    projectionMatrix = np.reshape(projectionMatrix, (4,4), 'F')
    image_parameters= [viewMatrix, projectionMatrix, img_size]

    rot_matrix = np.reshape(pb.getMatrixFromQuaternion(orientation), [3,3])
    T_matrix = np.zeros((4,4))
    T_matrix[3,3] = 1
    T_matrix[0:3,0:3] = rot_matrix
    T_matrix[0:3,3] = position
    
    u_grasps = grasp_univ_transformation(T_matrix, grasps)
    twoDgrasps = threeDtopixel_transformation(u_grasps, image_parameters)
    draw_realgrasps(u_grasps, new_scores)
    draw_grasps(twoDgrasps)
    return u_grasps
    
def pointpixel_transformation(points, image_parameters):
    view_matrix = image_parameters[0]
    projection_matrix = image_parameters[1]
    img_size = image_parameters[2]
    p = np.ones((4, len([points])))
    p[0:3,:] = points.T
    viewpos = view_matrix.dot(p)
    clippos = projection_matrix.dot(viewpos)
    normalize = clippos[0:3,:] / clippos[3,:]
    column = (normalize[0,:] + 1)*(img_size-1)/2
    row = (1 - normalize[1,:])*(img_size-1)/2
    px = np.array([column, row])
    return px.T
    
def threeDtopixel_transformation(grasps, image_parameters):
    view_matrix = image_parameters[0]
    projection_matrix = image_parameters[1]
    img_size = image_parameters[2]
    v1 = np.ones((4, len(grasps)))
    v2 = np.ones((4, len(grasps)))
    v1[0:3,:] = grasps[:,0,:].T
    v2[0:3,:] = grasps[:,1,:].T
    viewpos1 = view_matrix.dot(v1)
    viewpos2 = view_matrix.dot(v2)
    clippos1 = projection_matrix.dot(viewpos1)
    clippos2 = projection_matrix.dot(viewpos2)
    
    normalize1 = clippos1[0:3,:] / clippos1[3,:] #(3,m)
    normalize2 = clippos2[0:3,:] / clippos2[3,:]
    
    column1 = (normalize1[0,:] + 1)*(img_size-1)/2 #column index of contact point 1 in the image pixel
    column2 = (normalize2[0,:] + 1)*(img_size-1)/2 #column index of contact point 2 in the image pixel
    row1 = (1 - normalize1[1,:])*(img_size-1)/2 #row index of contact point 1 in the image pixel
    row2 = (1 - normalize2[1,:])*(img_size-1)/2 #row index of contact point 2 in the image pixel

    px1 = np.array([column1, row1]) #(2,m)
    px2 = np.array([column2, row2])
    twoD_grasps = np.hstack((px1.T, px2.T)).reshape(-1,2,2) #reshaped from (m,4) the grasp is represented in ((column, row), (column, row))
    return twoD_grasps
    
def draw_grasps (grasps):
    depth_im = Image.open('C:\\Users\\KosugeLab\\Desktop\\Adam\\Grasp_Planning\\Simulation\\Image\\depth.png')
    draw = ImageDraw.Draw(depth_im)
    i = 0
    while(i < len(grasps)):
        draw.line(
                (
                        (grasps[i,0,0], grasps[i,0,1]), (grasps[i,1,0], grasps[i,1,1])
                        ), 
                fill = (255,0,0)
                )
        i = i+1
        
    depth_im.show()
    


visualization('hv18')