import matplotlib.pyplot as plt

from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
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

from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import numpy as np


def computeKthNNProximityDistances(points, k_nn=5, write=False, filename="", density=0.0):
    """
    Compute a normalized proximity distance matrix using the distance to the k-th nearest neighbor.
    
    For each point a, let:
       kth_nn(a) = distance from a to its k-th nearest neighbor (using Euclidean distance).
    
    Then, for any two points a and b, define:
       norm_D(a,b) = Euclidean_distance(a,b) / ( kth_nn(a) + kth_nn(b) ).
    
    An edge will be included in the proximity graph if norm_D(a,b) < epsilon.
    
    Parameters:
      points  : list/array of 2D points.
      k_nn    : int, number of nearest neighbors to use (the k-th neighbor is used).
      write   : bool, if True, saves the distance matrix to a file.
      filename: str, file name to save the matrix if write==True.
      
    Returns:
      dist_mat : full symmetric normalized distance matrix.
      lower_tri: list of lists containing the lower-triangular part.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import pdist, squareform
    
    n = len(points)
    points_arr = np.array(points)
    # Build a KD-tree for fast neighbor queries.
    tree = cKDTree(points_arr)
    # Query k_nn+1 neighbors (first neighbor is the point itself with distance 0).
    distances, indices = tree.query(points_arr, k=k_nn+1)
    # kth_nn is the distance to the k-th nearest neighbor.
    kth_nn = distances[:, k_nn]  # shape: (n,)
    
    # Compute full pairwise Euclidean distance matrix.
    D = squareform(pdist(points_arr, metric='euclidean'))
    # Compute the sum of kth neighbor distances for each pair.
    sum_nn = kth_nn[:, None] + kth_nn[None, :]
    ratio_matrix = (np.maximum(kth_nn[:, None], kth_nn[None, :]) / 
                    np.minimum(kth_nn[:, None], kth_nn[None, :]))**density
    # Normalize the distance matrix.
    norm_D = D / sum_nn * ratio_matrix
    
    # Build lower triangular list.
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

def processStippleFile(filename, k_nn=5, epsilon = 1.0, density=0.0, shouldDraw = False, shouldDrawEdges = False):
    
    points, xs, ys = readStipplefile(filename)

    # compute the pair-wise distances 

    dist_mat, lower_tri = computeKthNNProximityDistances(points, k_nn=k_nn, filename= filename + "_distmat.txt", write = False, density=density)
    
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

def compute_cluster_density_terms(labels, n_components, avg_knn):
    """
    Compute a density term for each cluster as:
      density_term = (max(avg_knn in cluster)) / (min(avg_knn in cluster) + eps)
    """
    eps = 1e-9
    # Initialize with extremes
    max_d = np.full(n_components, -np.inf)
    min_d = np.full(n_components, np.inf)
    
    for i, lab in enumerate(labels):
        d = avg_knn[i]
        if d > max_d[lab]:
            max_d[lab] = d
        if d < min_d[lab]:
            min_d[lab] = d
            
    cluster_density = max_d / (min_d + eps)
    return cluster_density

def merge_clusters_by_density(labels, cluster_density, density_threshold=0.1):
    """
    Merge clusters whose density terms differ by less than density_threshold.
    
    Parameters:
      labels          : array of initial cluster labels (from connected_components)
      cluster_density : array of length n_components, density term per cluster.
      density_threshold : float, merging threshold.
    
    Returns:
      new_labels : merged cluster labels for each point.
    """
    n_components = len(cluster_density)
    parent = list(range(n_components))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(a, b):
        rootA = find(a)
        rootB = find(b)
        if rootA != rootB:
            parent[rootB] = rootA
            
    # Compare each pair of clusters and merge if density terms are similar.
    for c1 in range(n_components):
        for c2 in range(c1+1, n_components):
            if abs(cluster_density[c1] - cluster_density[c2]) < density_threshold:
                union(c1, c2)
    
    # Remap labels to consecutive numbers.
    new_labels = np.zeros_like(labels)
    cluster_map = {}
    new_label = 0
    for i, lab in enumerate(labels):
        root = find(lab)
        if root not in cluster_map:
            cluster_map[root] = new_label
            new_label += 1
        new_labels[i] = cluster_map[root]
    
    return new_labels


if __name__ == "__main__":

    # using argparse for argument handling
    parser = argparse.ArgumentParser(description="SING computation of given data.")
    parser.add_argument("--filename", type=str, help="The name of the file to process")
    parser.add_argument("--filetype", type=str, choices=["stipples", "species", "disks"], help="The type of the file (stipples, species, disks)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for SING")
    parser.add_argument("--drawEdges", type=bool, default=False, help="Draw SING edges")
    parser.add_argument("--k_nn", type=int, default=5, help="Number of nearest neighbors for distance computation")
    parser.add_argument("--density", type=float, default=0.0, help="density term for proximity metrics")
    args = parser.parse_args()

    filename = args.filename
    filetype = args.filetype
    epsilon = args.epsilon
    shouldDrawEdges = args.drawEdges
    k_nn = args.k_nn
    density = args.density

    shouldDraw = True

    random.seed(1335)

    distance_matrix = []    
    if filetype == "stipples":
        points, distance_matrix, lower_tri = processStippleFile(filename, k_nn=k_nn, density=density, epsilon=epsilon, shouldDraw = shouldDraw, shouldDrawEdges = shouldDrawEdges)
    elif filetype == "species":
        points, radii, distance_matrix, lower_tri = processDiskFile(filename, epsilon, shouldDrawEdges = shouldDrawEdges)
    elif filetype == "disks":
        points, distance_matrix, lower_tri = processBasicDiskFile(filename, epsilon, shouldDrawEdges = shouldDrawEdges)

    # plt.savefig(filename+str(epsilon)+"_basic.pdf",bbox_inches='tight', pad_inches=0)

    diag = compute_persistence_diagram(distance_matrix)

    barcode = gudhi.plot_persistence_barcode(diag)

    if(shouldDraw):
        plt, ax = createBasicPlot()
        threshold = epsilon
        edges, adj_mat = extractSINGEdges(distance_matrix, threshold)

        if(shouldDrawEdges):
            drawEdges(ax, edges, points)

        # extract the SING connected components
        n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)
        print(f"Initial number of components: {n_components}")
        
        # ----- New: Merge clusters by similar density term -----
        # Recompute the average kNN distance for each point.
        points_arr = np.array(points)
        tree = cKDTree(points_arr)
        distances, _ = tree.query(points_arr, k=k_nn+1)
        avg_knn = np.mean(distances[:,1:], axis=1)
        
        # Compute density term for each cluster.
        cluster_density = compute_cluster_density_terms(labels, n_components, avg_knn)
        # Merge clusters whose density terms differ by less than the threshold.
        merged_labels = merge_clusters_by_density(labels, cluster_density, density_threshold=0.6)
        n_merged = len(np.unique(merged_labels))
        print(f"Number of merged clusters: {n_merged}")
        # -------------------------------------------------------
        
        drawPoints(plt, points, n_merged, merged_labels, 2, ax)
        plt.show()



    