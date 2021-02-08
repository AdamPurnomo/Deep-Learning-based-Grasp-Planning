# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:23:27 2019

@author: Adam Syammas Zaki P
"""

friction_coef = 0.5

""" 
There is something important to note. Majority of image processor library are using (x,y)
coordinate system, while numpy works like a matrix where the coordinate of pixel is written in (row, colum)
or in another word (y,x). That is why on line 133 in this code, I deliberately exchange the indices of the grasp
candidates so that it is represented in the form of (x,y)
"""

# %%
import numpy as np
from scipy import ndimage as nd
import cv2
from PIL import Image, ImageDraw
import scipy.spatial.distance as ssd
import trimesh as tr 

    
def normals(depthImg, edgeImg):
    '''
    Calculate normal vectors of each grasp candidates
    #input
    depthImg    : depth image with data type uint8
    edgeImg     : edge image with binary data type

    #output
    normal      : normal vectors belong to edge pixels
                  Shape (n, 2)
    edge pixel  : list edge pixel from edge image
                  Shape (n, 2) 
    '''
    edge_pixels = np.where(edgeImg != [0])
    edge_pixels = np.c_[edge_pixels[0], edge_pixels[1]]

    #this is very important to note. Gradient image will only work if the data is in the form of float64
    depth = np.float64(depthImg) 
    sx = nd.sobel(depth, axis=0, mode= 'constant') #gradient in x direction
    sy = nd.sobel(depth, axis=1, mode='constant') #gradient in y direction
    
    normals = np.zeros((len(edge_pixels),2))
    
    for i, pix in enumerate(edge_pixels):
        dx = sx[pix[0], pix[1]]
        dy = sy[pix[0], pix[1]]
        #Negative signs are assigned so that the normal vectors are pointing outward
        normal_vec = np.array([-dx,-dy], dtype=float) 
        if np.linalg.norm(normal_vec) == 0:
            normal_vec = np.array([1,0])
        normal_vec = normal_vec / np.linalg.norm(normal_vec) 
        normals[i, :] = normal_vec
    return normals, edge_pixels

def antipodal_sampler (edge_pixels, normals, grasp_range):
    '''
    Antipodal grasp candidates sampling from depth. 
    #input
    edge_pixels  :  list of edge pixels from edge image
                    Shape (n, 2)
    normals      :  The corresponding normal vectors of edge pixels
                    Shape (n, 2)
    grasp_range  :  Maximum and minimum reach of end effector grasp

    #output
    grasps      : Set of antipodal grasp candidates
                  Shape (n, 2, 2)
    normal      : Corresponding normal vector of each grasp candidates
                  Shape (n, 2, 2)
    ''' 
    min_dist = grasp_range[0]
    max_dist = grasp_range[1]
    
    dists = ssd.squareform(ssd.pdist(edge_pixels))
    normals_ip = normals.dot(normals.T)
    
    valid_pairs = np.where(
            (normals_ip < -0.9)&
            (dists < max_dist) &
            (dists > min_dist)
            )

    valid_pairs = np.c_[valid_pairs[0], valid_pairs[1]]

    contact_points1 = edge_pixels[valid_pairs[:,0], :]
    contact_points2 = edge_pixels[valid_pairs[:,1], :]
    normals_points1 = normals[valid_pairs[:,0], :]
    normals_points2 = normals[valid_pairs[:,1], :]
    
    v = contact_points1 - contact_points2
    v_norm = np.linalg.norm(v, axis=1)
    v = v / np.tile(v_norm[:, np.newaxis], [1, 2])
    ip1 = np.sum(normals_points1 * v, axis=1)
    ip2 = np.sum(normals_points2 * (-v), axis=1)
    antipodal_indices = np.where((ip1<-np.cos(np.arctan(friction_coef)))
                                &(ip2<-np.cos(np.arctan(friction_coef))))[0]
    grasp_indices = np.random.choice(antipodal_indices, size = antipodal_indices.shape[0], replace = False)

    k = 0
    grasps = []
    normals = []
    while k < grasp_indices.shape[0] and len(grasps) < 80:
        p1 = contact_points1[grasp_indices[k],:]
        p2 = contact_points2[grasp_indices[k],:]
        n1 = normals_points1[grasp_indices[k],:]
        n2 = normals_points2[grasp_indices[k],:]
        
        grasps.append([p1, p2])
        normals.append([n1, n2])
        k = k+1
        
    grasps = np.asarray(grasps)
    normals = np.asarray(normals)
        
    g_zeros = np.zeros(grasps.shape)
    if(grasps.size !=0):
        g_zeros[:,:,0] = grasps[:,:,1]
        g_zeros[:,:,1] = grasps[:,:,0]
        grasps = g_zeros.astype(int)
    
    return grasps, normals

def sample_surface(mesh, count):
    """
    Sample the surface of a mesh, returning the specified
    number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    #input
    mesh    : trimesh.Trimesh Geometry to sample the surface of
    count   : int Number of points to return

	#output
    samples : (count, 3) float Points in space on the surface of mesh
    face_index : (count,) int Indices of faces for each sampled point
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
    
def volume_antipodal_sampler(stl, grasp_range):
    '''
    Antipodal grasp candidates sampling from 3D cad object.
    #input
    stl          :  STL file of the particular object
    grasp_range  :  Maximum and minimum reach of end effector grasp

    #output
    grasps       : Set of antipodal grasp candidates
                  Shape (n, 2, 3)
    normal       : Corresponding normal vector of each grasp candidates
                  Shape (n, 2, 3)
    face_indices : face indices of each grasp candidate
                   shape (n, 2, 1)
    ''' 
    
    mesh = tr.load_mesh(stl)
    mesh.remove_degenerate_faces
    mesh.vertices -= mesh.center_mass
    samples, indices= sample_surface(mesh, 1000)
    normals = mesh.face_normals[indices]
    distance = ssd.squareform(ssd.pdist(samples))
    normals_ip = normals.dot(normals.T)
    valid_pairs = np.where((normals_ip < -0.85) & 
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

    antipodal_indices = np.where((ip1>np.cos(np.arctan(friction_coef)))&(ip2>np.cos(np.arctan(friction_coef))))[0]
    
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
    


