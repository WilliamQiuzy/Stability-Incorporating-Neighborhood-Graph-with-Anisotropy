import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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

from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import numpy as np

def compute_angle(points, k_nn=3, max_k=7, variance_threshold=0.1):
    """
    Compute the principal direction angle for each point by dynamically extending the neighborhood,
    inspired by ellipsoid enlargement, using PCA from scikit-learn.

    Parameters:
      points           : list of 2D points.
      k_start          : int, initial number of nearest neighbors (excluding the point itself).
      max_k            : int, maximum number of neighbors to consider.
      variance_threshold : float, threshold for variance ratio to stop neighborhood extension.

    Returns:
      angles : numpy array of shape (n,), with each value in [0, pi).
    """
    n = len(points)
    points_arr = np.array(points)
    tree = cKDTree(points_arr)
    angles = np.zeros(n)
    
    for i in range(n):
        k = k_nn
        while k <= max_k:
            # Query k+1 neighbors (including the point itself)
            _, indices = tree.query(points_arr[i], k=k+1)
            # Exclude the point itself
            neighbors = points_arr[indices[1:]]
            # Perform PCA on 2 components to assess variance
            pca = PCA(n_components=2)
            pca.fit(neighbors)
            # Get explained variances (eigenvalues)
            lambda1, lambda2 = pca.explained_variance_
            # Compute variance ratio: fraction of variance in minor direction
            variance_ratio = lambda1 / (lambda1 + lambda2) if (lambda1 + lambda2) > 0 else 0
            # Stop if variance ratio is below threshold (sufficient anisotropy captured)
            # or if max_k is reached
            if variance_ratio > variance_threshold or k == max_k:
                # Use the first principal component for the angle
                principal = pca.components_[0]
                theta = np.arctan2(principal[1], principal[0])
                if theta < 0:
                    theta += np.pi
                angles[i] = theta
                break
            k += 1  # Add one more neighbor
    return angles



def computeAvgKNNProximityDistances(points, k_nn=5, write=False, filename="", density=0.0, gamma=0.5):
    n = len(points)
    points_arr = np.array(points)
    # Build KD-tree for fast neighbor queries.
    tree = cKDTree(points_arr)
    # Query k_nn+1 nearest neighbors (first is self with distance 0).
    distances, indices = tree.query(points_arr, k=k_nn+1)
    # Compute average distance (excluding self).
    avg_knn = np.mean(distances[:, 1:], axis=1)  # shape (n,)
    
    # Compute principal direction angles for each point.
    angles = compute_angle(points, k_nn=k_nn)  # array of shape (n,), in [0, pi)
    
    # Compute full pairwise Euclidean distance matrix.
    D = squareform(pdist(points_arr, metric='euclidean'))
    # Compute initial normalized distance: d_norm = D / (knn(a) + knn(b))
    sum_knn = avg_knn[:, None] + avg_knn[None, :]
    norm_D = D / sum_knn
    
    # Compute the density ratio for each pair:
    # R(a,b) = (max(knn(a), knn(b)) / min(knn(a), knn(b)))^(density)
    ratio_matrix = (np.maximum(avg_knn[:, None], avg_knn[None, :]) / 
                    np.minimum(avg_knn[:, None], avg_knn[None, :]))**density
    
    angle_diff = np.abs(angles[:, None] - angles[None, :])
    angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
    anisotropic_multiplier = gamma + (1 - gamma) * np.sin(angle_diff)
    
    # Final normalized distance matrix:
    final_norm_D = norm_D * ratio_matrix * anisotropic_multiplier
    # final_norm_D = norm_D * anisotropic_multiplier
    
    # Build lower triangular list.
    lower_tri = [list(final_norm_D[i, :i]) for i in range(n)]
    
    if write:
        np.savetxt(filename, final_norm_D, delimiter=",")
    
    return final_norm_D, lower_tri


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

def processStippleFile(filename, k_nn=5, epsilon = 1.0, density=0.0, gamma=0.5, useGamma=False, shouldDraw = False, shouldDrawEdges = False):
    
    points, xs, ys = readStipplefile(filename)

    points_arr = np.array(points)

    # Pick a reference point index
    ref_idx = 100

    # Run the analysis
    analyzeReferencePointDistanceWithAxis(points_arr, ref_idx=ref_idx, k_nn=5, density=0.0, gamma=0.5)

    # compute the pair-wise distances 

    dist_mat, lower_tri = computeAvgKNNProximityDistances(points, k_nn=k_nn, filename= filename + "_distmat.txt", write = False, density=density, gamma=gamma)
    
    # visualisation for the points, the clusters, and possibly the edges

    if(shouldDraw):
        plt, ax = createBasicPlot()
        threshold = gamma if useGamma else epsilon
        edges, adj_mat = extractSINGEdges(dist_mat, threshold)

        if(shouldDrawEdges):
            drawEdges(ax, edges, points)

        # extract the SING connected components

        n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)
        print(f"Number of components: {n_components}")
        
        drawPoints(plt, points, n_components, labels, 2, ax)

        # save_labels(filename+"_SING.txt", points, labels)
        

    return points, dist_mat, lower_tri

def analyzeReferencePointDistanceWithAxis(points, ref_idx=0, k_nn=5, density=0.0, gamma=0.5):
    """
    1. Compute distance matrix using your custom metric (with gamma, angle-based, etc.).
    2. Visualize distances from the reference point, plus the principal axis among its k_nn neighbors.
    """
    # Suppose we have your existing function:
    # dist_mat, lower_tri = computeAvgKNNProximityDistances(...)
    dist_mat, lower_tri = computeAvgKNNProximityDistances(
        points, 
        k_nn=k_nn, 
        write=False, 
        filename="",
        density=density, 
        gamma=gamma
    )
    
    # Now plot
    plotReferenceDistancesWithAxis(points, dist_mat, ref_idx=ref_idx, k_nn=k_nn,
                                   title=f"Distances + Axis (ref={ref_idx}, gamma={gamma})")

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
    parser.add_argument("--xaxis", type=str, choices=["epsilon", "gamma"], default="epsilon", 
                        help="Choose x-axis variable for TDA (epsilon or gamma)")
    parser.add_argument("--gamma", type=float, default=0.5, 
                        help="Gamma value for the angle term in distance computation and as TDA x-axis if selected")
    args = parser.parse_args()

    filename = args.filename
    filetype = args.filetype
    epsilon = args.epsilon
    shouldDrawEdges = args.drawEdges
    k_nn = args.k_nn
    density = args.density
    gamma = args.gamma
    useGamma = (args.xaxis == "gamma")

    shouldDraw = True

    random.seed(1335)

    distance_matrix = []    
    if filetype == "stipples":
        points, dist_mat, lower_tri = processStippleFile(filename, k_nn=k_nn, density=density, epsilon=epsilon, shouldDraw = shouldDraw, shouldDrawEdges = shouldDrawEdges, gamma=gamma, useGamma=useGamma)
    elif filetype == "species":
        points, radii, dist_mat, lower_tri = processDiskFile(filename, epsilon, shouldDrawEdges = shouldDrawEdges)
    elif filetype == "disks":
        points, dist_mat, lower_tri = processBasicDiskFile(filename, epsilon, shouldDrawEdges = shouldDrawEdges)

    # plt.savefig(filename+str(epsilon)+"_basic.pdf",bbox_inches='tight', pad_inches=0)

    if useGamma:
        diag = compute_persistence_diagram(distance_matrix, max_edge=gamma)
    else:
        diag = compute_persistence_diagram(distance_matrix, max_edge=epsilon)

    barcode = gudhi.plot_persistence_barcode(diag)

    if(shouldDraw):
        plt, ax = createBasicPlot()
        threshold = gamma if useGamma else epsilon
        edges, adj_mat = extractSINGEdges(dist_mat, threshold)

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
        merged_labels = merge_clusters_by_density(labels, cluster_density, density_threshold=0.2)
        n_merged = len(np.unique(merged_labels))
        print(f"Number of merged clusters: {n_merged}")
        # -------------------------------------------------------
        
        drawPoints(plt, points, n_merged, merged_labels, 2, ax)
        plt.show()


    