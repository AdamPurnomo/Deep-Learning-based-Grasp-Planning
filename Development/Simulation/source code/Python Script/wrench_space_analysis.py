import numpy as np
import scipy.spatial as sp
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

#initializing constant parameters
friction_coef = 1
n_cone = 6
local_com = np.array([0,0,0])
origin_ws = np.array([0,0,0,0,0,0])

def removNestings(l):
    output = []
    for i in l: 
        if type(i) == list: 
            removNestings(i) 
        else: 
            output.append(i) 

def torque_scale(bb_extend):
    '''
    Determine the constant to scale torque wrenches so that it is invariant
    relative to the size of the object. 
    #input
    bb_extent : The size of object bounding box
                Shape (3, )
    #output
    t_scal : scaling factor for torque vectors
             Shape (1, )
    '''
    t_scal = 1 / np.median(bb_extend)
    return t_scal

def in_hull(p, cloud):
    """
    Test if points in `p` are in `hull`
    This function is not actually needed, just for debugging.
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `cloud` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    #input
    p   : point that we want to test 
          Shape (1, dim)
    cloud : set of points that form convex hull
            Shape (n, dim)

    #output
    bool : Whether the point lies inside the convex hull or not
    """
    if not isinstance(cloud,Delaunay):
        hull = Delaunay(cloud, qhull_options = 'Qt QJ')

    return hull.find_simplex(p)>=0

def min_distance_hull (p, cloud):
    """
    Calculate the closest distance from a point to the convex hull facets.
    'p' is the particular point that we want to calculate, should be an array of K-dimension,
    cloud is the points that make up the hull (N points x K dimension).

    #input
     p      : point that we want to test. Normally this is the origin of hull 
             Shape (1, dim)
    cloud   : set of points that form convex hull
              Shape (n, dim)
    
    #output
    q       : The closest normalized distance from test point to hull facets
    index   : index of facet which is closest from test point
    proj_p  : the exact location of the closest distance to hull facets
    """
    if not isinstance(cloud,ConvexHull):
        hull = ConvexHull(cloud, qhull_options = 'Qx')

    dist = []
    proj_p = [] #list of projected point on the hyperplanes
    
    for i in range(len(hull.equations)):
        #return to the normal vector of the hyperplane
        n  = np.array(hull.equations[i,0:len(hull.equations[i])-1]) 
        #signed distance from test point to the hyperplane depending on the normal vector
        s = np.dot(n, p) + hull.equations[i,len(hull.equations[i])-1] 
        #vector from projected point on the hyperplane to the test point
        s_vec = s * n 
        #projected point on the hyperplane
        q = p - s_vec 
        q = np.hstack(q)
    
        dist.append(abs(s))
        proj_p.append(q)
    #if the distance is positive, the point is outside the convex hull    
    if(s > 0): 
        q = 0
    else:
        q = min(dist) / max(dist)

    return q, dist.index(min(dist)), proj_p

def evaluate_grasps(grasps, normals, com, t_scal):
    '''
    Evaluating grasp candidates with Ferrary-Canny grasp evaluation metrics

    #input
    grasps  : Set of grasp candidates
              Shape (n, 2, 3)
    normals : The corresponding normal vectors associated with eahc grasp candidates
              Shape (n, 2, 3)
    com     : Center of mass of the object in object local coordinate
              Usually at the origin
              Shape (3, )
    t_scal  : A constant to scale the torque

    #output
    grasp_scores    : score evaluation of each grasp candidates
                      Shape (n, )
    invalid_indexes : index of grasp candidates whose convex hull of the wrenches
                      cannot be computed
    '''
    
    def cone_wrenches(grasp, normal):
        '''
        Generating cloud wrenches from a single grasp candidates

        #input
        grasp   : A single grasp candidate
                  Shape (1, 2, 3)
        normal  : normal vector of the grasp candidate
                  Shape (1, 2, 3)

        #output
        cloud_w : Set of wrenches generated from 
                  couloumb friction with soft finger model
                  Shape ((2*n_cone+1),  6)  
        '''
        cloud_w = [] #set wrenches
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
            
            for j in range (n_cone):
                #Cone Direction
                ei = h*n + r*np.cos(j*(2*np.pi/n_cone))*e1 + r*np.sin(j*(2*np.pi/n_cone))*e2 
                #constant for linear convex summation
                alpha = 1 / (len(grasp)*n_cone) 
                #Projection of normal force along the cone extrema
                force = alpha*ei 
                torque = t_scal*np.cross(contact_p-com,force)
                #generating wrenches 
                wrenches = np.hstack([force, torque]) 
                cloud_w.append(wrenches)
                #additional wrenches for soft contact model for each contact point
                if (j == n_cone-1):
                    cloud_w.append(np.array([0,0,0,alpha*n[0],alpha*n[1],alpha*n[2]])) 
                
        cloud_w = np.reshape(cloud_w, (2*(n_cone+1), 6)) 
        return cloud_w
    
    grasp_scores = []
    l_scores = []
    invalid_indexes = []
    for i in range (len(grasps)):
        try:
            center = (grasps[i,0] + grasps[i,1]) / 2
            relative_dist = np.linalg.norm(center - local_com)*t_scal 
            loc_s = 1 - relative_dist
        
            cloud_w = cone_wrenches(grasps[i], normals[i])
            score, hyperp_index, closest_loc = min_distance_hull(origin_ws, cloud_w)
            grasp_scores.append(score)
            l_scores.append(loc_s)
        except(sp.qhull.QhullError):
            invalid_indexes.append(i)
            pass
    grasp_scores = np.asarray(grasp_scores)    
    grasp_scores = grasp_scores / max(grasp_scores)
    l_scores = np.asarray(l_scores)
    return grasp_scores, l_scores, np.asarray(invalid_indexes)

def viz_3D_hull (points, test_point):
    '''
    Visualizing convex hull in 3D space
    #input
    points  : all points included in the convex hull
    test point  : the point to be tested whether it is inside or outside the hull

    #output
    None
    '''
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