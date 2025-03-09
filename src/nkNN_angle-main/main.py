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

def compute_angle(points, k_nn=5):
    """
    Compute the principal direction angle for each point based on its k_nn nearest neighbors,
    using PCA from scikit-learn.

    Parameters:
      points : list of 2D points.
      k_nn   : int, number of nearest neighbors (excluding the point itself) to use.

    Returns:
      angles : numpy array of shape (n,), with each value in [0, pi).
    """
    n = len(points)
    points_arr = np.array(points)
    tree = cKDTree(points_arr)
    # Query k_nn+1 neighbors (first neighbor is the point itself)
    _, indices = tree.query(points_arr, k=k_nn+1)
    angles = np.zeros(n)
    
    # Create a PCA object; we only need the first component.
    pca = PCA(n_components=1)
    
    for i in range(n):
        # Exclude the point itself
        neighbor_idx = indices[i][:]
        neighbors = points_arr[neighbor_idx]
        # Center the data (PCA in sklearn automatically centers the data)
        pca.fit(neighbors)
        principal = pca.components_[0]
        theta = np.arctan2(principal[1], principal[0])
        if theta < 0:
            theta += np.pi
        angles[i] = theta
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
        points, distance_matrix, lower_tri = processStippleFile(filename, k_nn=k_nn, density=density, epsilon=epsilon, shouldDraw = shouldDraw, shouldDrawEdges = shouldDrawEdges, gamma=gamma, useGamma=useGamma)
    elif filetype == "species":
        points, radii, distance_matrix, lower_tri = processDiskFile(filename, epsilon, shouldDrawEdges = shouldDrawEdges)
    elif filetype == "disks":
        points, distance_matrix, lower_tri = processBasicDiskFile(filename, epsilon, shouldDrawEdges = shouldDrawEdges)

    # plt.savefig(filename+str(epsilon)+"_basic.pdf",bbox_inches='tight', pad_inches=0)

    if useGamma:
        diag = compute_persistence_diagram(distance_matrix, max_edge=gamma)
    else:
        diag = compute_persistence_diagram(distance_matrix, max_edge=epsilon)

    barcode = gudhi.plot_persistence_barcode(diag)

    if shouldDraw:
        plt.show()



    