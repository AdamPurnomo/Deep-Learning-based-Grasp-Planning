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
import socket
import pandas as pd

from model import Discriminator, MaskPredictor, GraspNet

#parameters
parameters = {'hv18': [35, 45, 220, 100], 
              'hv8':[170, 180, 220, 220],
	      'hv9': [35, 45, 220, 100]}
img_save_dir = r'../Grasp Sequence Image/ '
temp_dir = r'../Temporary Data/ '

data_size = parameters['hv9'][2:4]
grasp_range = parameters['hv9'][0:2]
img_size = [1024,1280]
roi = [230,730,200,970]

#client communication
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1234))
s.listen(5)

#building and loading  model's weight
weight_dir0 = r'../Weights/Discriminator_hv9'
weight_dir1 = r'../Weights/MaskPredictor_hv9'

D = Discriminator()
D.build((1,data_size[0],data_size[1],3))
D.load_weights(weight_dir0)

M = MaskPredictor()
M.build((1,data_size[0],data_size[1],3))
M.load_weights(weight_dir1)

model = GraspNet()
model.build((1,data_size[0],data_size[1],3))

#combining weights
for i in range(11):
    weight = D.layers[i].get_weights()
    model.layers[i].set_weights(weight)

for i in range(11,20):
    weight = M.layers[i-4].get_weights()
    model.layers[i].set_weights(weight)

scores_list = []
masks_list = []
confidence = []

@tf.function
def inference(image):
    score, mask = model(image)
    return score, mask

iterator = 0
loop_num = pd.read_csv(r'..\..\system_loop\loop_num.txt', header=None)[0][0]

while(iterator<loop_num):
	#reading flag
	print("listening . . .")
	clientsocket, address = s.accept()
	print(f"Connection from {address} has been establised.")
	action_done_flag = int(clientsocket.recv(1024).decode("utf-8"))
	clientsocket.close()
	print("Action Done")		

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
		cart_apvect = pe.apv_to_cartesian(pointmap, ga_vect, p_grasps, 35)
		R, euler = pe.extract_pose_v2(p_grasps, cart_apvect, 35)
		euler = euler*180/np.pi
		

		#image sequence logging
		save_dir = img_save_dir + str(iterator) + '.png' 
		index = np.random.randint(0,3)
		best_grasp = p_grasps[index]
		ap_vect = ga_vect[index]
		orient = euler[index]
		rcapvect = cart_apvect[index]
		confidence.append(p_scores[index])
		
		
		#Drawing Rectangle
		ip.draw_best_grasp(best_grasp, ap_vect, depthImg, data_size, save_dir)
		print("Image has been saved!\n")

		#grasp position and orientation logging
		c = lp.to_cartesian(best_grasp, pointmap)
		cap_vect = lp.to_cartesian(ap_vect, pointmap, False)

		home_base = np.array([0, 400, 401.5])
		c = c - home_base
		angle = np.pi*orient/180
		rel_pre_grasp = pe.pre_grasp_pos_v2(c, angle,30)
		stroke = c - rel_pre_grasp

		orient = np.array([-orient[0], orient[1], -orient[2]])
		sixdfirst_pose = np.array([rel_pre_grasp[0], rel_pre_grasp[1] ,rel_pre_grasp[2], orient[0], orient[1], orient[2]]).reshape((1,6))
		sixd_stroke = np.array([stroke[0], stroke[1], stroke[2], 0, 0, 0]).reshape((1,6))

		fourdfirst_pose = np.array([c[0], c[1], 0, 0, 0, orient[2]]).reshape((1,6))	
		fourd_stroke = np.array([0, 0, c[2], 0, 0, 0]).reshape((1,6))		

		file_dir = temp_dir + r'sixdfirst_pose.txt'
		np.savetxt(file_dir, sixdfirst_pose, delimiter=' ', fmt='%f')
		file_dir = temp_dir + r'sixd_stroke.txt'
		np.savetxt(file_dir, sixd_stroke, delimiter=' ', fmt='%f')
		
		file_dir = temp_dir + r'fourdfirst_pose.txt'
		np.savetxt(file_dir, fourdfirst_pose, delimiter=' ', fmt='%f')
		file_dir = temp_dir + r'fourd_stroke.txt'
		np.savetxt(file_dir, fourd_stroke, delimiter=' ', fmt='%f')


		print("Pose has been saved!\n")
		
		#flag
		clientsocket, addres = s.accept()
		clientsocket.send(bytes("0", "utf-8"))
		print("Action Start")
		clientsocket.close() 

		iterator += 1

confidence = np.array(confidence)
file_dir = temp_dir + r'confidence level.txt'
np.savetxt(file_dir, confidence, delimiter = ' ', fmt = '%f')


		


