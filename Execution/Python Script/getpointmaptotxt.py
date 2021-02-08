import sys
sys.path.append(r"C:/Users/Adam/Desktop/Controller_Ver.3.8.7/Themes/adam/Execution/Camera Python Binding/x64/Release")

savedir = r'temporary/'
import getpointmap
import numpy as np

def importpointmap():
    for i in range(1,65):
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

