# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:35:20 2019

@author: Adam Syammas Zaki P
"""
file_loc = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\data\model\_object.stl'
import numpy as np
import trimesh as tr

mesh = tr.load_mesh(file_loc, process = False)
mesh.show()