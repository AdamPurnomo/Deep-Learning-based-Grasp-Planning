import sys
sys.path.append(r"../Camera Python Binding/x64/Release")

import getpointmap
import numpy as np

def importpointmap(iteration,savedir):
	'''
    Collecting pointmap data from camera to txt files.
    
    #input
    iteration   : Number of pointmap data taken
    savedir     : Directory to save the pointmap data

    #output
    None.
    '''
    for i in range(iteration):
        print('Iteration : ' + str(i))
        pointmap = getpointmap.getpointmap()
        pointmap = np.array(pointmap)
        xindices = np.arange(0, len(pointmap), 3)
        yindices = xindices + 1
        zindices = yindices + 1
        x = pointmap[xindices]
        y = pointmap[yindices]
        z = pointmap[zindices]
        filedir = savedir + r'pointmap' + str(i)
        np.save(filedir, z)

