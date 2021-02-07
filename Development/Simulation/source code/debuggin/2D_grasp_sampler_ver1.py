import numpy as np
import cv2
import scipy.spatial.distance as ssd
from PIL import Image, ImageDraw
from scipy import ndimage as nd


friction_coef = 2.0
max_dist = 15
min_dist = 0

def draw_normals(grasps, normals, depth_im):
    draw = ImageDraw.Draw(depth_im)
    i = 0
    for i in range (1):
        draw.line(
                (grasps[i][0][1], grasps[i][0][0], grasps[i][0][1] + 5*normals[i][0][1], grasps[i][0][0] + 5*normals[i][0][0]), fill = (255,0,0)
                       )
                
        draw.line(
                (grasps[i][1][1], grasps[i][1][0], grasps[i][1][1] + 5*normals[i][1][1], grasps[i][1][0] + 5*normals[i][1][0]), fill = (255,0,0)
                )
                
    depth_im.save(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\grasp_normal_debug.png')

def surface_normals(depth_im, edge_pixels):
    depth_im = np.float64(depth_im)
    sx = nd.sobel(depth_im,axis = 0, mode = 'constant')
    sy = nd.sobel (depth_im, axis = 1, mode = 'constant')
    #grad = np.gradient(depth_im) #calculate the gradient of depth image. Read : gradient of image
    normals = np.zeros((len(edge_pixels), 2))
    
    for i, pix in enumerate(edge_pixels):
        dx = sx[pix[0], pix[1]]
        dy = sy[pix[0], pix[1]]
        normal_vec = np.array([dx,dy], dtype=float) #input normal vector of each egde pixel
        if np.linalg.norm(normal_vec) == 0:
            normal_vec = np.array([1,0])
        normal_vec = normal_vec / np.linalg.norm(normal_vec) #normalize the normal vector 
        normals[i, :] = normal_vec #append normal vector to an array so that it has the same index as the edge pixel
        
    return normals

def antipodal_grasp_sampler(depth_im, edge_pixels):
    edge_normal = surface_normals(depth_im, edge_pixels) #compute normal vector of the edge image
    normal_ip = edge_normal.dot(edge_normal.T) #inner product of normal vector of every edge pixels (MxM) with M is the length of edgepixel
    dists = ssd.squareform(ssd.pdist(edge_pixels)) #distance between each pixel, (MxM), has the same index as normal ip
    
    #np.savetxt("normal_ip.csv", normal_ip, delimiter = ",")
    #np.savetxt("pair_distance.csv", dists, delimiter = ",")
    
    #filtering indices form normal_ip and dists which satisfy the characteristics of antipodal grasps
    valid_pairs = np.where(
            (normal_ip < -0.9)&
            (dists < max_dist) &
            (dists > min_dist)
            )
    valid_pairs = np.c_[valid_pairs[0], valid_pairs[1]] #set of pair indices in edgepixels which are possibly antipodal point
    
    contact_points1 = edge_pixels[valid_pairs[:,0], :]
    contact_points2 = edge_pixels[valid_pairs[:,1], :]
    contact_normals1 = edge_normal[valid_pairs[:,0], :]
    contact_normals2 = edge_normal[valid_pairs[:,1], :]
    
    grasps = [contact_points1, contact_points2]
    normals = [contact_normals1, contact_normals2]
    """
    v = contact_points1 - contact_points2
    v_norm = np.linalg.norm(v, axis=1)
    v = v / np.tile(v_norm[:, np.newaxis], [1, 2])
    ip1 = np.sum(contact_normals1 * v, axis=1)
    ip2 = np.sum(contact_normals2 * (-v), axis=1)
    ip1[ip1 > 1.0] = 1.0
    ip1[ip1 < -1.0] = -1.0
    ip2[ip2 > 1.0] = 1.0
    ip2[ip2 < -1.0] = -1.0
    beta1 = np.arccos(ip1)
    beta2 = np.arccos(ip2)
    alpha = np.arctan(friction_coef)
    antipodal_indices = np.where((beta1 < alpha) & (beta2 < alpha))[0]
    grasp_indices = np.random.choice(antipodal_indices,
                                         size=antipodal_indices.shape[0],
                                         replace=False)
    k = 0
    grasps = []
    normals = []
    while k < antipodal_indices.shape[0] and len(grasps) < 40:
            grasp_ind = grasp_indices[k]
            p1 = contact_points1[grasp_ind, :]
            p2 = contact_points2[grasp_ind, :]
            n1 = contact_normals1[grasp_ind, :]
            n2 = contact_normals2[grasp_ind, :]
            #            width = np.linalg.norm(p1 - p2)
            k += 1
            grasps.append([p1,p2])
            normals.append([n1,n2])
     """       
    return grasps, normals

def draw_image (grasps, depth_im):
    draw = ImageDraw.Draw(depth_im)
    for i in range (1):
        draw.line((grasps[i][0][1], grasps[i][0][0], grasps[i][1][1], grasps[i][1][0]), fill = (255,0,0))
        
    #draw.line((0,0,224,0), fill = (255,0,0))
    depth_im.save(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\grasps.png')


depth_image = r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\depth_cropped.png'
image = cv2.imread(depth_image, cv2.IMREAD_GRAYSCALE)
image_filt = cv2.GaussianBlur(image, (7,7), 0)
edge_im = cv2.Canny(image_filt, 10, 50)
cv2.imwrite(r'C:\Users\Adam Syammas Zaki P\Documents\Assignment\Grasp Planning\Simulation\Image\5x5gaussian_edge.jpg', edge_im)

edge_pixels = np.where(edge_im == 255)
edge_pixels = np.c_[edge_pixels[0], edge_pixels[1]]

grasps, normals = antipodal_grasp_sampler(image, edge_pixels)

depth_im = Image.open(depth_image)
draw_image(grasps, depth_im)
draw_normals(grasps, normals, depth_im)


#cv2.imshow('image',image)
#cv2.waitKey()
r"""
I = np.array([
        [1,0,0,0],
        [1,1,0,0],
        [1,1,1,0],
        [1,1,1,1]
        ])
G = np.gradient(I)
print(G[0])
print(G[1])"""
