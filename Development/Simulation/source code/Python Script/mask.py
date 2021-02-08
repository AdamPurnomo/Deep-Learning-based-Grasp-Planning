import numpy as np 
import pybullet as pb 
import cv2
from PIL import Image, ImageDraw

def z_vec_transformation(rot_matrix, grasp_candidates):
    '''
    This function perform transformation of z-axis aligned
    vector (grasp approaching vector) from local coordinate to universal coodinate
    according to the orientation of each object in the simulation.
    The transformed z_vec corresponding to each object are then 
    associated with each corresponing grasp candidates ID
   
    #input
    orientation : The object's rotation matrix 
                  Shape (3,3)
    grasp_candidates : 4D tensor representing grasp candidates.
                       Shape (n, 2, 3)

    #output
    z_vec       : list of transformed z_vector from local coordinates to 
                  universal coordinate.
                  Shape (n, 3)
    '''

    #z-axis aligned vector in local coordinate of the object
    local_vector1 = np.array([0,0,0.25])
    local_vector2 = np.array([0,0,-0.25])
    rotated_vec = np.array([rot_matrix.dot(local_vector1), rot_matrix.dot(local_vector2)])
    z = rotated_vec[:, 2]
    index = np.argmax(z)
    rotated_vec = rotated_vec[index]
    grasp_centers = (grasp_candidates[:,0,:] + grasp_candidates[:,1,:])/2
    z_vec = grasp_centers + rotated_vec
    
    return z_vec

def to_pixel(points, rendering_matrices, img_size):
    '''
    This function performs projection of 3D vector into
    2D images.

    #input
    points : set of 3D vector 
             Shape (n, 3)
    rendering_matrices  : set of matrices utilized for image rendering 
                          which consists as follows
                            - view matrix
                            - projection matrix
    img_size            : image size of projected scene
                          Shape (2,)

    #output 
    px : set of 2D vector (pixel) representing the projection
         of 3D vectors into an image.
         Shape (n, 2)

    '''
    view_matrix = rendering_matrices[0]
    projection_matrix = rendering_matrices[1]
    row_size, column_size = img_size
    p = np.ones((4, points.shape[0]))
    p[0:3,:] = points.T
    viewpos = view_matrix.dot(p)
    clippos = projection_matrix.dot(viewpos)
    normalize = clippos[0:3,:] / clippos[3,:]
    column = (normalize[0,:] + 1)*(column_size-1)/2
    row = (1 - normalize[1,:])*(row_size-1)/2
    px = np.array([column, row]).astype(int)
    return px.T

def mask_visualization(p_zvec, grasps, data_size):
    '''
    Create a visualization for orientation mask. The visualization will not
    be used for training the network, instead they are only used for the sake
    of visualization.
    
    #input
    p_zvec      : projection of z_vec in the image that represents object orientation
                  Shape (n, 2)
    p_grasps    : projection of grasps in the image
                  Shape (n, 2, 2)
    data_size   : the size of training data image we want to create.
                  the size will depend on the size of the object
    save_dir    : directory to save the data
    data_size    : size of data representation

    #Output
    None
    This function will save the data to the corresponding directory
    '''
    viz = []
    vector = []
    v_mat = []
    s_mat = []
    invalid_indices = []
    height = data_size[0]
    width = data_size[1]
    
    for i in range(len(grasps)):
        #generating initial mask
        mask = np.zeros((height, width, 3)).astype(np.uint8)       

        #rotation transformation
        center = ((grasps[i,0] + grasps[i,1]) / 2).astype(int)
        vect1 = grasps[i,1] - grasps[i,0]
        vect2 = grasps[i,0] - grasps[i,1]
        angle1 = np.arctan2(vect1[1], vect1[0]) * 180 / np.pi
        angle2 = np.arctan2(vect2[1], vect2[0]) * 180 / np.pi
        angle = np.array([angle1, angle2]) 
        indices = np.argmin(np.abs(angle))
        angle = angle[indices]
        M = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1.0)

        
        try:
            #creating mask
            z_vec = np.array([p_zvec[i, 0], p_zvec[i, 1], 1])
            z_vec = M.dot(z_vec)
            relative_vector = z_vec - center
            new_center = np.array([(width/2), (height/2)]).astype(int)
            new_zvec = (new_center + relative_vector).astype(int)

            im_pil = Image.fromarray(mask)
            draw = ImageDraw.Draw(im_pil)
            draw.ellipse(((new_zvec[0]-3, new_zvec[1]-3), (new_zvec[0]+3, new_zvec[1]+3)), fill =(0,0,255))
            mask = np.array(im_pil)
            indices = np.where(mask == 255)
            vect_mat = np.zeros((height, width))
            vect_mat[indices[0], indices[1]] = 1

            sparse_mat = np.zeros((height, width))
            sparse_mat[new_zvec[1], new_zvec[0]] = 1

    
            viz.append(mask)
            vector.append(relative_vector)
            v_mat.append(vect_mat)
            s_mat.append(sparse_mat)
        except(IndexError):
            invalid_indices.append(i)
    return viz, vector, v_mat, s_mat, invalid_indices

def save_mask(basedir, vector, v_mat, s_mat, scores, nameid):
    '''
    save mask representation
    #input      : base directory to save the data
    vector      : grasp approaching vector 
                  Shape (3,)
    v_mat       : Projection of grasp approaching vector represented as line
                  on images
                  Shape (m,n) matrix
    s_mat       : Projection of grasp approaching vector represented as dots
                  on images
                  Shape (m,n) matrix
    Scores      : grasping score of the grasp candidate associated with the grasp approaching vector
    nameid      : name id for the grasp candidate and the grasp approaching vector

    #output
    None
    '''
    data_size = len(nameid)
    for i in range(data_size):
        if(scores[i,0] > 0.7 and scores[i,1] > 0.9 and scores[i,2] == 0):
            root = basedir
            name = nameid[i]
            vector_arr = root + r'\vector\ ' + name +'.npy'
            np.save(vector_arr, vector[i])

            vmat_arr = root + r'\vector matrix\ ' + name +'.npy'
            np.save(vmat_arr, v_mat[i])

            smat_arr = root + r'\sparse matrix\ ' + name +'.npy'
            np.save(smat_arr, s_mat[i])






        




        
