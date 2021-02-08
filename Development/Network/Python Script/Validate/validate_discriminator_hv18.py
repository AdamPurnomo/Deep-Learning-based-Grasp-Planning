# %%
import sys 
sys.path.append(r'../../../utility')
sys.path.append(r'../model')
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import antipodal_sampler as ap
import img_processor as ip  
from model import Discriminator


parameters = {'hv18': [35, 45, 0.02, 0.025, 220, 100], 
              'hv8':[38, 44, 0.09, 0.11, 220, 220],
              'hv9' : [35, 45, 0.02, 0.025, 220, 100]}

#setting parameters
grasp_range = parameters['hv18'][0:2]
data_size = parameters['hv18'][4:6]

#loading pointmap from temporary folder
pointmap = np.loadtxt(r"../../Test Data/hv18 test/pointmap00.txt")
xindices = np.arange(0, len(pointmap), 3)
yindices = xindices + 1
zindices = yindices + 1

x = pointmap[xindices]
y = pointmap[yindices]
z = pointmap[zindices]

depthMap = z.reshape((1024,1280))
depthMapROI = depthMap[230:730, 200:970]

#generating depth image from depth map
depthImg = cv2.normalize(depthMapROI,
                        dst=None,
                        alpha=0,
                        beta=255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8UC1)
depthImg = 255-depthImg

#filling missing values from depth map
mask = np.zeros(depthImg.shape).astype('uint8')
indices = np.where(depthImg == 255)
mask[indices] = 1
depthImg = cv2.inpaint(depthImg, mask, 3, cv2.INPAINT_TELEA)

#computing edge image
blur = cv2.GaussianBlur(depthImg, (7,7), 0)
edgeImg = cv2.Canny(blur, 5, 60)

#grasps sampling
normals, edgepx = ap.normals(depthImg, edgeImg)
grasps, normals = ap.antipodal_sampler(edgepx, normals, grasp_range)

#generating grasp image representations
data, bb = ip.data_generator(grasps, depthImg, data_size)
data = np.array(data, dtype='float32') / 255.

#draw grasp
graspImg = ip.draw_grasps(grasps, depthImg, rgb=(0,0,255))
# %%

#inference
bs = 1
model = Discriminator()
model.build((bs, 220, 100, 3))
model.load_weights(r"../../Output/Weights/Discriminator_hv18")
prediction = []
@tf.function
def validation_step(model, image):
    prediction = model(image)
    return prediction

for image in data:
       image = np.expand_dims(image, axis = 0)
       pred = validation_step(model, image)
       prediction.append(pred)

prediction = np.array(prediction)
prediction = np.vstack(prediction)
prediction = np.hstack(prediction)

p_indices = np.where(prediction>0.5)
n_indices = np.where(prediction<=0.5)

p_grasps = grasps[p_indices]
n_grasps = grasps[n_indices]

pgraspImg = ip.draw_grasps(p_grasps, depthImg, rgb=(0,255,0))
ngraspImg = ip.draw_grasps(n_grasps, depthImg, rgb=(255,0,0))

import matplotlib.pyplot as plt 

plt.imshow(graspImg)
plt.show()

plt.imshow(pgraspImg)
plt.show()

plt.imshow(ngraspImg)
plt.show()




# %%
