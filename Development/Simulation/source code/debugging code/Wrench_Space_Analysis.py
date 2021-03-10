import numpy as np
import scipy.spatial as sp
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from itertools import product, combinations

friction_coef = 0.5
n_cone = 6
t_scal = 1/0.1
local_com = np.array([0,0,0])
origin_ws = np.array([0,0,0,0,0,0])

def removNestings(l):
    output = []
    for i in l: 
        if type(i) == list: 
            removNestings(i) 
        else: 
            output.append(i) 

def torque_scale(grasps, com):
    center = (grasps[:, 0] + grasps[:, 1]) / 2
    com = np.tile(com, (len(center), 1))
    r_dist = np.linalg.norm(center - com, axis=1)
    t_scal = 1 / max(r_dist)
    return t_scal

def in_hull(p, cloud):
    """
    Test if points in `p` are in `hull`
    This function is not actually needed, just for debugging.
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `cloud` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(cloud,Delaunay):
        hull = Delaunay(cloud, qhull_options = 'Qt QJ')

    return hull.find_simplex(p)>=0

def min_distance_hull (p, cloud):
    """
Calculate the closest distance from a point to the convex hull facets.
'p' is the particular point that we want to calculate, should be an array of K-dimension,
cloud is the points that make up the hull (N points x K dimension).
"""

    #if (cloud.shape[1] > 3):
     #   model = PCA(n_components = cloud.shape[1]-1).fit(cloud)
      #  proj_cloud = model.transform(cloud)

    if not isinstance(cloud,ConvexHull):
        hull = ConvexHull(cloud, qhull_options = 'Qx')

    dist = []
    proj_p = [] #list of projected point on the hyperplanes
    
    for i in range(len(hull.equations)):
        n  = np.array(hull.equations[i,0:len(hull.equations[i])-1]) #return to the normal vector of the hyperplane
        s = np.dot(n, p) + hull.equations[i,len(hull.equations[i])-1]  #signed distance from test point to the hyperplane depending on the normal vector
        s_vec = s * n #vector from projected point on the hyperplane to the test point
        q = p - s_vec #projected point on the hyperplane
        q = np.hstack(q)
    
        dist.append(abs(s))
        proj_p.append(q)
        
    if(s > 0):  #if the distance is positive, the point is outside the convex hull
        q = 0
    else:
        q = min(dist) / max(dist)

    return q, dist.index(min(dist)), proj_p

def viz_3D_hull (points, test_point):
    hull  = ConvexHull(points, qhull_options = 'QJ')
    min_d, index, q = min_distance_hull(test_point, points)

    fig = plt.figure()
    ax  = fig.add_subplot(1,2,1, projection='3d')
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])
        ax.plot(points[simplex, 0], points[simplex, 1],points[simplex, 2], 'k-')

    for i in range (len(q)):
        x = np.array([test_point[0],q[i][0]])
        y = np.array([test_point[1],q[i][1]])
        z = np.array([test_point[2],q[i][2]])
        ax.scatter(q[i][0], q[i][1],q[i][2], c = 'r', marker = 'x')
        ax.plot(x, y, z, 'r--')

    ax.scatter(test_point[0], test_point[1], test_point[2], r'o')

    ax  = fig.add_subplot(1,2,2, projection='3d')
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])
        ax.plot(points[simplex, 0], points[simplex, 1],points[simplex, 2], 'k-')

    for p in points:
        ax.scatter(p[0], p[1], p[2], c = 'b' ,marker ='o')
        
    x = np.array([test_point[0],q[index][0]])
    y = np.array([test_point[1],q[index][1]])
    z = np.array([test_point[2],q[index][2]])
    ax.scatter(q[index][0], q[index][1],q[index][2], c = 'r', marker = 'x')
    ax.plot(x, y, z, 'r--')
    ax.scatter(test_point[0], test_point[1], test_point[2], c = 'b' ,marker ='o')

    plt.show()
   

          
def evaluate_2d(grasps, normals, com):
    t_scal = torque_scale(grasps, com)
    
    def twoD_wrenches(grasp, normal):
        cloud_w = []
        for i in range(len(grasp)):
            contact_p = grasp[i]
            n = normal[i] / np.linalg.norm(normal[i])
            
            if(n[0] != 0 and n[1] != 0):
                s = np.array([1/n[0], -1/n[1]])
                s = s / np.linalg.norm(s)
            else:
                s = np.array([0,0])
                index = np.where(s == 0)
                s[index] = 1
                
            
            h = 1 / (np.sqrt(1 + np.square(friction_coef)))
            r = h * friction_coef
            
            for j in range(n_cone):
                ei = h*n + r*np.cos(j*(2*np.pi/n_cone))
                alpha = 1 / (len(grasp)*n_cone) #constant for linear convex summation
                force = alpha*ei #Projection of normal force along the cone extrema
                torque = t_scal*np.cross(contact_p-com,force)
                wrenches = np.hstack([force, torque]) #generating wrenches
                cloud_w.append(wrenches)
        cloud_w = np.reshape(cloud_w, (2*n_cone, 3))
        return cloud_w  
    
    
    
    
    #grasp_scores = []
    loc_scores = []
    for i in range(len(grasps)):
        center = ((grasps[i,0] + grasps[i,1]) / 2)
        relative_dist = np.linalg.norm(center - com)*t_scal 
        loc_s = 1 - relative_dist
        loc_scores.append(loc_s)
        
           
        #cloud_w = twoD_wrenches(grasps[i], normals[i])
        #score, hyperp_index, closest_loc = min_distance_hull(origin_ws, cloud_w)
        #grasp_scores.append(score)
    
    #grasp_scores = np.asarray(grasp_scores)
    loc_scores = np.asarray(loc_scores)
    #grasp_scores = grasp_scores / max(grasp_scores)
    grasp_scores = loc_scores
    return grasp_scores
    
    

def evaluate_grasps(grasps, normals, com):
    
    t_scal = torque_scale(grasps, com)
    
    def cone_wrenches(grasp, normal):
        cloud_w = [] #set wrenches
        #vis_f = []
        #tau = [] #set of torque
        #f = [] #set of force
        for i in range (len(grasp)):
            contact_p = grasp[i]
            n = normal[i]
            
            a = n[0]
            b = n[1]
            c = n[2]
            
            e1 = np.array([c-b, a-c, b-a])
            e1 = e1 / np.linalg.norm(e1)
            
            e2 = np.cross(n, e1)
            e2 = e2 / np.linalg.norm(e2)
            
            h = 1 / (np.sqrt(1 + np.square(friction_coef)))
            r = h * friction_coef
            
            
            #f = []
            for j in range (n_cone):
                ei = h*n + r*np.cos(j*(2*np.pi/n_cone))*e1 + r*np.sin(j*(2*np.pi/n_cone))*e2 #Cone Direction
                alpha = 1 / (len(grasp)*n_cone) #constant for linear convex summation
                force = alpha*ei #Projection of normal force along the cone extrema
                torque = t_scal*np.cross(contact_p-com,force)
                wrenches = np.hstack([force, torque]) #generating wrenches 
                cloud_w.append(wrenches)
                if (j == n_cone-1):
                    cloud_w.append(np.array([0,0,0,alpha*n[0],alpha*n[1],alpha*n[2]])) #additional wrenches for soft contact model for each contact point
                #f.append(force)
                #force_debug.append(force)
                #tau.append(torque)
                #vis_f.append(f)
        #g_force = [0,0,-9.8]
        #g_torque = [0,0,0]
        #g_wrench = np.hstack([g_force, g_torque])
        #cloud_w.append(g_wrench)
        cloud_w = np.reshape(cloud_w, (2*(n_cone+1), 6)) 
        #tau = np.reshape(tau, (2*n_cone, 3))
        #f = np.reshape(f, (2*n_cone, 3))
        #force_debug = np.reshape(force_debug, (2*n_cone, 3))
        return cloud_w

    
    grasp_scores = []
    loc_scores = []
    invalid_indexes = []
    for i in range (len(grasps)):
        try:
            center = (grasps[i,0] + grasps[i,1]) / 2
            relative_dist = np.linalg.norm(center - local_com)*t_scal 
            loc_s = 1 - relative_dist
        
            cloud_w = cone_wrenches(grasps[i], normals[i])
            score, hyperp_index, closest_loc = min_distance_hull(origin_ws, cloud_w)
            grasp_scores.append(score)
            loc_scores.append(loc_s)
        except(sp.qhull.QhullError):
            invalid_indexes.append(i)
            pass
        
        
    grasp_scores = np.asarray(grasp_scores)
    loc_scores = np.asarray(loc_scores)
    
    grasp_scores = grasp_scores / max(grasp_scores)
    #grasp_scores = grasp_scores
    return grasp_scores, np.asarray(invalid_indexes), loc_scores

"""
#Debugging Code
#Testing minimum distance and visualizing 2D convex hull, worked"
points = np.random.rand(30,2)
test_point = np.array([0.5, 0.5])
hull  = ConvexHull(points)
min_d, index, q = min_distance_hull(test_point, points)
plt.subplot(121)
plt.plot(points[:,0], points[:,1], "o")
for i in range (len(q)):
    x = np.array([test_point[0],q[i][0]])
    y = np.array([test_point[1],q[i][1]])
    plt.plot(q[i][0], q[i][1], "rx")
    plt.plot(x, y, "r--")
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    
plt.subplot(122)
plt.plot(points[:,0], points[:,1], "o")
plt.plot(test_point[0], test_point[1], "ro")
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
x = np.array([test_point[0],q[index][0]])
y = np.array([test_point[1],q[index][1]])
plt.plot(q[index][0], q[index][1], "rx")
plt.plot(x, y, "r--")
plt.show()
"""

r""" 
### Debug 3D Grasps
for k in range(10):
    com = np.array([0,0,0])
    grasp = np.array([[-10, 0, k], [10, 0, k]])
    normal = np.array([[1, 0, 0], [-1, 0, 0]])
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    r = [-10, 10]
    ax.plot([grasp[0][0],grasp[1][0]],[grasp[0][1],grasp[1][1]], [grasp[0][2],grasp[1][2]], 'k-')
    
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="b")
            
    cloud_w, force_cone, torque, force = cone_wrenches(grasp, normal, com)
    p = np.array([0,0,0,0,0,0])
    q, index, proj = min_distance_hull(p, cloud_w)
    
    for i in range(len(force_cone)):
        for j in range(len(force_cone[i])):
            ax.quiver(grasp[i][0], grasp[i][1], grasp[i][2],force_cone[i][j][0],force_cone[i][j][1], force_cone[i][j][2],length=50)
    
            
    
    ax.set_title('Grasp Score = %f' %q)
    plt.show()
    print("Grasp Score = ", q)
p = np.array([0,0,0])
#min_f, index_f, q_f = min_distance_hull(p, force)
#min_t, index_t, q_t = min_distance_hull(p, torque)
#viz_3D_hull(force, p)
#viz_3D_hull(torque, p)
#plt.show()
"""
"""
### Debug 2D Grasps
cube = np.array([
    [-10, 10],
    [10, 10],
    [10, -10],
    [-10, -10],
    [-10, 10]
    ])
com = np.array([0,0])
grasp1 = np.array([
    [5,10],
    [8,-10]
    ])
grasp2 = np.array([
    [0,10],
    [0,-10]
    ])
grasp3 = np.array([
    [8,10],
    [-8,-10]
    ])
theta = np.pi*45/180
f_c1 = np.array([
    [-np.cos(theta), -np.sin(theta)],
    [np.cos(theta), -np.sin(theta)]
    ])
f_c2 = np.array([
    [-np.cos(theta), np.sin(theta)],
    [np.cos(theta), np.sin(theta)]
    ])
f_c = [f_c1, f_c2]
wrenches = []
for i in range (2):
    for j in range (2):
        force = f_c[i][j]
        torque = 0.1*np.cross(grasp1[i], force)
        wrench = np.hstack([force, torque])
        wrenches.append(wrench)
points = np.reshape(wrenches, [4, 3])
test_point = np.array([0,0,0])
hull  = ConvexHull(wrenches)
min_d, index, q = min_distance_hull(test_point, points)
fig = plt.figure()
ax  = fig.add_subplot(1,3,1)
ax.scatter(com[0],com[1], c = 'r', marker = 'o')
for i in range (len(cube)-1):
    ax.plot(cube[i], cube[i+1], 'k-')
    
for i in range(len(f_c)):
    for j in range(len(f_c[i])):
        ax.quiver(grasp1[i][0], grasp1[i][1],
                  10*f_c[i][j][0],10*f_c[i][j][1], color = 'b')
ax  = fig.add_subplot(1,3,2, projection='3d')
for simplex in hull.simplices:
    simplex = np.append(simplex, simplex[0])
    ax.plot(points[simplex, 0], points[simplex, 1],points[simplex, 2], 'k-')
for i in range (len(q)):
    x = np.array([test_point[0],q[i][0]])
    y = np.array([test_point[1],q[i][1]])
    z = np.array([test_point[2],q[i][2]])
    ax.scatter(q[i][0], q[i][1],q[i][2], c = 'r', marker = 'x')
    ax.plot(x, y, z, 'r--')
ax.scatter(test_point[0], test_point[1], test_point[2], r'o')
ax  = fig.add_subplot(1,3,3, projection='3d')
ax.set_title('Grasp Score = %f' %min_d)
for simplex in hull.simplices:
    simplex = np.append(simplex, simplex[0])
    ax.plot(points[simplex, 0], points[simplex, 1],points[simplex, 2], 'k-')
x = np.array([test_point[0],q[index][0]])
y = np.array([test_point[1],q[index][1]])
z = np.array([test_point[2],q[index][2]])
ax.scatter(q[index][0], q[index][1],q[index][2], c = 'r', marker = 'x')
ax.plot(x, y, z, 'r--')
ax.scatter(test_point[0], test_point[1], test_point[2], r'o')
plt.show()
"""