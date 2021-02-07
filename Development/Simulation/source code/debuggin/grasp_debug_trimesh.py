# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:40:55 2019

@author: Adam Syammas Zaki P
"""

import numpy as np
import trimesh as tr



def ray_tracing(local_grasps):
    stl = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\data\model\_object.stl'
    mesh = tr.load_mesh(stl, process = 'false')
    size = mesh.bounding_box.primitive.extents
    
    ray_origins = local_grasps[:,0]
    ray_directions = local_grasps[:,1]
    locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins, ray_directions)
    ray_visualize = tr.load_path(np.hstack((ray_origins, ray_origins + ray_directions*2.0)).reshape(-1, 2, 3))
    
    mesh.unmerge_vertices()
    mesh.visual.face_colors = [255,255,255,255]
    mesh.visual.face_colors[index_tri] = [255,0,0,255]
    
    scene = tr.Scene([mesh, ray_visualize])
    scene.show("gl")