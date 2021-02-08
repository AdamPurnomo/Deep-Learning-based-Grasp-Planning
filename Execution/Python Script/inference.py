# %%
import sys 
sys.path.append(r'../../Development/utility')
sys.path.append(r'../../Development/Network/Python Script/model')
import tensorflow as tf 
import numpy as np 
import cv2
import os
import random as r
import loadpointmap as lp
import img_processor as ip
import antipodal_sampler as ap 
import pose_estimator as pe 
import pandas as pd

from model import Discriminator, MaskPredictor, GraspNet

#parameters
parameters = {'hv18': [35, 45, 220, 100], 
              'hv8':[38, 44, 220, 220]}
file_dir = r'../Temporary Data/pointmap.txt'
img_save_dir = r'../Grasp Sequence Image/ '
pose_dir = r'../Temporary Data/pose.txt'

data_size = parameters['hv18'][2:4]
grasp_range = parameters['hv18'][0:2]
img_size = [1024,1280]
roi = [230,730,200,970]

#building and loading  model's weight
weight_dir0 = r'../Weights/Discriminator_hv18'
weight_dir1 = r'../Weights/MaskPredictor_hv18_v1'

D = Discriminator()
D.build((1,220,100,3))
D.load_weights(weight_dir0)

M = MaskPredictor()
M.build((1,220,100,3))
M.load_weights(weight_dir1)

model = GraspNet()
model.build((1,220,100,3))

#combining weights
for i in range(11):
    weight = D.layers[i].get_weights()
    model.layers[i].set_weights(weight)

for i in range(11,20):
    weight = M.layers[i-4].get_weights()
    model.layers[i].set_weights(weight)

scores_list = []
masks_list = []

@tf.function
def inference(image):
    score, mask = model(image)
    return score, mask

finished_flag = 0
iterator = 0

while(finished_flag == 0):
	#reading flag
	finished_flag = pd.read_csv(r'..\Temporary Data\finished_flag.txt', header=None)[0][0]
	action_done_flag = pd.read_csv(r'..\Temporary Data\action_done_flag.txt', header=None)[0][0]

	if(action_done_flag==1):
		#get pointmap and depth image
		pointmap = lp.load_pointmap(img_size, roi)
		depthImg = lp.to_depthImg(pointmap[2])

		#computing edge image
		blur = cv2.GaussianBlur(depthImg, (7,7), 0)
		edgeImg = cv2.Canny(blur, 5, 60)

		#grasps sampling
		normals, edgepx = ap.normals(depthImg, edgeImg)
		grasps, normals = ap.line_antipodal_sampler(edgepx, normals, grasp_range)

		#generating grasp image representations
		data, r_matrix = ip.data_generator(grasps, depthImg, data_size)
		data = np.array(data, dtype='float32') / 255.

		#inference
		i = 0
		scores_list.clear()
		masks_list.clear()
		for image in data:
			image = np.expand_dims(image, axis=0)
			tf.print('Inference Step: ', i)
			score, mask = inference(image)
			scores_list.append(score.numpy()[0])
			masks_list.append(mask.numpy()[0,:,:,0])
			i += 1
		
		scores = np.vstack(scores_list)
		masks = np.array(masks_list)

		#sorting positive candidates
		p_indices = np.where(scores >= 0.5)
		p_scores = scores[p_indices[0]]
		p_grasps = grasps[p_indices[0]]
		p_masks = masks[p_indices[0]]
		r_matrix = r_matrix[p_indices[0]]
		p_data = data[p_indices[0]]

		sort_indices = np.argsort(p_scores[:,0])[::-1]
		p_scores = p_scores[sort_indices]
		p_grasps = p_grasps[sort_indices]
		p_masks = p_masks[sort_indices]
		r_matrix = r_matrix[sort_indices]
		p_data = p_data[sort_indices]

		#extracting pose
		rel_vect = pe.locate_sparse_matrix(p_masks)
		ga_vect = pe.rotate_back(rel_vect, r_matrix, p_grasps)
		cart_apvect = pe.to_cartesian(pointmap, ga_vect, p_grasps, 40)
		R, euler = pe.extract_pose(p_grasps, cart_apvect, 40)
		euler = euler*180/np.pi

		#flag logging
		action_done_flag = np.array([0], dtype='int')
		np.savetxt(r'..\Temporary Data\action_done_flag.txt',action_done_flag)

		#image sequence logging
		save_dir = img_save_dir + str(iterator) + '.png' 
		best_grasp = p_grasps[0]
		ap_vect = ga_vect[0]
		orient = euler[0]
		dummy = ip.draw_grasp_representation(best_grasp, best=True, ap_vect, labels = 1, depthImg, data_size, save_dir)
		print("Image has been saved!\n")

		#grasp position and orientation logging
		c = lp.to_cartesian(best_grasp, pointmap)
		pose = np.hstack((c, orient))
		np.savetxt(pose_dir, pose, delimiter='\n')
		print("Pose has been saved!\n")

		iterator += 1


		


