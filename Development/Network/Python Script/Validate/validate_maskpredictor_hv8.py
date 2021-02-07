# %%
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import antipodal_sampler as ap
import img_processor as ip  
from model import MaskPredictor
import os
import random as r 


parameters = {'hv18': [35, 45, 0.02, 0.025, 220, 100], 
              'hv8':[170, 180, 0.09, 0.11, 220, 220]}

#setting parameters
grasp_range = parameters['hv8'][0:2]
data_size = parameters['hv8'][4:6]

#loading pointmap from temporary folder
pointmap = np.loadtxt(r"../../Test Data/Point Map Data/hv8/pointmap0.txt")
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
real_data, bb = ip.data_generator(grasps, depthImg, data_size)
rea_data = np.array(real_data, dtype='float32') / 255.

#draw grasp
graspImg = ip.draw_grasps(grasps, depthImg, rgb=(0,0,255))

#loading synthetic data
data_dir = r"../../Test Data/Mask/hv8"

file_path = []
for (root, dirs, files) in os.walk(data_dir):
    for file_name in files:
        path = os.path.join(root,file_name)
        list_form = path.split(os.path.sep)
        if(list_form[-2] == 'full'):
            file_path.append(path)
r.shuffle(file_path)

synthetic_data = []
sparse_matrix = []
vector_matrix = []
for im_path in file_path:
    image = cv2.imread(im_path)

    list_form = im_path.split(os.path.sep)
    file_name = list_form[-1]
    ext = file_name.split(sep='.')[1]
    name = file_name.split(sep='.')[0]
    
    #sparse matrix
    label_name = name + '.npy'
    label_path = list_form
    label_path[-1] = label_name
    label_path[-2] = 'sparse matrix'
    label_path = os.path.sep.join(label_path)
    label = np.load(label_path)
    synthetic_data.append(image)
    sparse_matrix.append(label)

synthetic_data = np.array(synthetic_data, dtype='float32') / 255.
sparse_matrix = np.array(sparse_matrix, dtype='float32')
# %%
#inference
bs = 1
model = MaskPredictor()
model.build((bs, 220, 220, 3))
model.load_weights(r"../../Training Output/Weights/MaskPredictor_hv8")
s_prediction = []
r_prediction = []
@tf.function
def validation_step(model, image):
    prediction = model(image)
    return prediction

for image in synthetic_data:
       image = np.expand_dims(image, axis = 0)
       pred = validation_step(model, image)
       s_prediction.append(pred)

for image in rea_data:
    image = np.expand_dims(image, axis=0)
    pred = validation_step(model, image)
    r_prediction.append(pred)





# %%
