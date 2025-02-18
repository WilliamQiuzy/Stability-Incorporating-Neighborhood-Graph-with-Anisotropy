import matplotlib.pyplot as plt

from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import gudhi
from compare import *
from input import *
from draw import *
from utils import *
import argparse

import random


# core distance
def core_dist(points, k):
    tree = cKDTree(points)
    dists = []
    for point in points:
        ds, inds = tree.query(point, k)
        dists.append(ds[-1])
    return dists

def computeAvgKNNProximityDistances(points, k_nn=5, write=False, filename=""):
    """
    Compute a normalized proximity distance matrix using an average kNN estimator.
    
    For each point a, we define:
    
         knn(a) = average_{i=1}^{k_nn} (distance from a to its i-th nearest neighbor)
    
    Then, for any two points a and b, the normalized distance is computed as:
    
         d_norm(a,b) = Euclidean_distance(a,b) / (knn(a) + knn(b))
    
    Parameters:
      points : list or array of 2D points.
      k_nn   : int, number of nearest neighbors to use.
      write  : bool, if True, the full distance matrix is saved to a file.
      filename: str, file name to save if write==True.
      
    Returns:
      dist_mat : full symmetric normalized distance matrix.
      lower_tri: list of lists containing the lower-triangular part (for persistence computation).
    """
    n = len(points)
    points_arr = np.array(points)
    # Build a KD-tree for fast neighbor queries.
    tree = cKDTree(points_arr)
    # Query k_nn+1 neighbors for each point (the first neighbor is the point itself with distance 0).
    distances, indices = tree.query(points_arr, k=k_nn+1)
    # Average the distances from index 1 to k_nn (exclude the self-distance at index 0).
    avg_knn = np.mean(distances[:, 1:], axis=1)  # shape: (n,)
    
    # Compute full pairwise Euclidean distance matrix.
    D = squareform(pdist(points_arr, metric='euclidean'))
    # Compute pairwise sum of the knn estimates.
    sum_knn = avg_knn[:, None] + avg_knn[None, :]
    # Compute normalized distance matrix.
    norm_D = D / sum_knn
    
    # Build lower-triangular list.
    lower_tri = [list(norm_D[i, :i]) for i in range(n)]
    
    if write:
        np.savetxt(filename, norm_D, delimiter=",")
    
    return norm_D, lower_tri

##### circle metric SING computation
def computeSINGCircleDistances(points, radii, write=False, filename = ""):

    nn_dists = np.full(len(points), np.inf)
    
    dist_mat = np.zeros((len(points), len(points)))


    # find nearest neighbor for each point
    for i, point in enumerate(points):
        for j in range(0,i):
            dist = distF(point, points[j], radii[i], radii[j])

            if dist < nn_dists[i]:
                nn_dists[i] = dist

            if dist < nn_dists[j]:
                nn_dists[j] = dist

    # compute the lower triangular matrix containing the pairwise distances, needed for the
    # persistence computation
    lower_tri = []
    for i, point in enumerate(points):
        dists = []
        for j in range(0,i):
            distance = distF(point, points[j], radii[i], radii[j])/ (nn_dists[i] + nn_dists[j])
            dist_mat[i][j] = distance
            dist_mat[j][i] = distance
            dists.append(distance)
        lower_tri.append(dists)

    if write:
        np.savetxt(filename, dist_mat, delimiter=",")  

    
    return dist_mat, lower_tri


# extract the SING edges for connected components and drawing

def extractSINGEdges(dist_mat, epsilon = 1.0):
    edges = []
    adj_mat = np.zeros((len(dist_mat[0]), len(dist_mat[0])))

    # for each pair of points, check the SING condition

    for i in range(0,len(dist_mat[0])):
        for j in range(0,i):
            if(dist_mat[i][j] <= epsilon):
                
                edges.append((i,j))
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1
    
    return edges, adj_mat



# process species file, with coordinates and radius, ignore extra information per point

def processDiskFile(filename, epsilon = 1.0, shouldDraw = True, shouldDrawEdges = False):
    
    # keep coordinates and radii, disregard extra information, not needed for our application

    data, species, xs, ys, radii, colours, points, species_labels = readDiskFile(filename)    
    
    # use the circle distance

    dist_mat, lower_tri = computeSINGCircleDistances(points, radii)

    # drawing function for visualisation, of the clusters and possibly the SING edges

    if(shouldDraw):
        plt, ax = createPlot(0,0,10000,10000)

        edges, adj_mat = extractSINGEdges(dist_mat, epsilon)

        if(shouldDrawEdges):
            drawEdges(ax, edges, points)

        # extract the connected components of the SING graph

        n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)
        print(f"Number of components: {n_components}")
        drawCirclesPerClassNoSpecies(data, points, n_components, labels, ax)

    return points, radii, dist_mat, lower_tri


# process basic disk file, with coordinates and radius only

def processBasicDiskFile(filename, epsilon = 1.0, shouldDraw = True, shouldDrawEdges = False):
    data, xs, ys, radii, points = readBasicDiskFile(filename) 

    # compute the pairwise distances, using the circle distance

    dist_mat, lower_tri = computeSINGCircleDistances(points, radii)

    # drawing the clusters and possibly the SING edges for visualisation purposes

    if(shouldDraw):
        plt, ax = createBasicPlot()
        ax.set_aspect("equal")
        edges, adj_mat = extractSINGEdges(dist_mat, epsilon)

        if(shouldDrawEdges):
            drawEdges(ax, edges, points)

        # extract the connected components from the SING graph

        n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)

        print(f"Number of components: {n_components}")
        
        drawCirclesPerClassNoSpecies(data, points, n_components, labels, ax, 0)

    return points, dist_mat, lower_tri



# process stipple file, that only contains 2D coordinates

def processStippleFile(filename, k_nn=5, epsilon = 1.0, shouldDraw = False, shouldDrawEdges = False):
    
    points, xs, ys = readStipplefile(filename)

    # compute the pair-wise distances 

    dist_mat, lower_tri = computeAvgKNNProximityDistances(points, k_nn=k_nn, filename= filename + "_distmat.txt", write = False)
    
    # visualisation for the points, the clusters, and possibly the edges

    if(shouldDraw):
        plt, ax = createBasicPlot()
        edges, adj_mat = extractSINGEdges(dist_mat, epsilon)

        if(shouldDrawEdges):
            drawEdges(ax, edges, points)

        # extract the SING connected components

        n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)
        print(f"Number of components: {n_components}")
        
        drawPoints(plt, points, n_components, labels, 2, ax)

        # save_labels(filename+"_SING.txt", points, labels)
        

    return points, dist_mat, lower_tri




if __name__ == "__main__":

    # using argparse for argument handling
    parser = argparse.ArgumentParser(description="SING computation of given data.")
    parser.add_argument("--filename", type=str, help="The name of the file to process")
    parser.add_argument("--filetype", type=str, choices=["stipples", "species", "disks"], help="The type of the file (stipples, species, disks)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for SING")
    parser.add_argument("--drawEdges", type=bool, default=False, help="Draw SING edges")
    parser.add_argument("--k_nn", type=int, default=5, help="Number of nearest neighbors for distance computation")
    args = parser.parse_args()

    filename = args.filename
    filetype = args.filetype
    epsilon = args.epsilon
    shouldDrawEdges = args.drawEdges
    k_nn = args.k_nn

    shouldDraw = True

    random.seed(1335)

    distance_matrix = []    
    if filetype == "stipples":
        points, distance_matrix, lower_tri = processStippleFile(filename, k=k_nn, epsilon=epsilon, shouldDraw = shouldDraw, shouldDrawEdges = shouldDrawEdges)
    elif filetype == "species":
        points, radii, distance_matrix, lower_tri = processDiskFile(filename, epsilon, shouldDrawEdges = shouldDrawEdges)
    elif filetype == "disks":
        points, distance_matrix, lower_tri = processBasicDiskFile(filename, epsilon, shouldDrawEdges = shouldDrawEdges)

    # plt.savefig(filename+str(epsilon)+"_basic.pdf",bbox_inches='tight', pad_inches=0)

    diag = compute_persistence_diagram(distance_matrix)

    barcode = gudhi.plot_persistence_barcode(diag)

    if shouldDraw:
        plt.show()



    