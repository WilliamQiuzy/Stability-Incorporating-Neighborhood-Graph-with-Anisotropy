#!/usr/bin/env python3
import argparse
import random
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import gudhi
from compare import *
from input import *
from draw import *
from utils import *


def computeKDEProximityDistances(points, bandwidth=None, write=False, filename=""):
    n = len(points)
    # Prepare data for KDE: shape should be (d, n)
    data = np.array(points).T  # For 2D, shape = (2, n)
    kde = gaussian_kde(data, bw_method=bandwidth)
    densities = kde.evaluate(data)  # density at each point, shape (n,)
    eps_small = 1e-10
    # Define density proxy as the reciprocal of the density estimate.
    density_est = 1.0 / (densities + eps_small)
    
    dist_mat = np.zeros((n, n))
    lower_tri = []
    for i, point in enumerate(points):
        dists = []
        for j in range(0, i):
            d = euclidean_distance(point, points[j])
            norm_d = d / (density_est[i] + density_est[j])
            dist_mat[i][j] = norm_d
            dist_mat[j][i] = norm_d
            dists.append(norm_d)
        lower_tri.append(dists)
    
    if write:
        np.savetxt(filename, dist_mat, delimiter=",")
    
    return dist_mat, lower_tri

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

def processStippleFile(filename, bandwidth=None, epsilon = 1.0, density = 0.0, shouldDraw = False, shouldDrawEdges = False):
    
    points, xs, ys = readStipplefile(filename)

    # compute the pair-wise distances 

    bandwidth1 = compute_bandwidth(points)

    dist_mat, lower_tri = computeKDEProximityDistances(points, bandwidth=bandwidth1, filename= filename + "_distmat.txt", write = False)
    
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

import numpy as np

def compute_bandwidth(points):
    """
    Compute the bandwidth for a set of 2D points using Silverman's rule.
    
    Parameters:
      points : array-like of shape (n_points, 2)
      
    Returns:
      h : float, the estimated bandwidth.
    
    Silverman's rule for 2D:
        h = sigma * n^{-1/6},
    where sigma is the average standard deviation in x and y.
    """
    pts = np.array(points)
    n = pts.shape[0]
    # Compute the standard deviation for each dimension (using ddof=1 for an unbiased estimator)
    stds = np.std(pts, axis=0, ddof=1)
    # Use the average standard deviation as the overall scale
    sigma = np.mean(stds)
    h = sigma * n**(-1/6)
    return h

# Example usage:
if __name__ == "__main__":
    # Generate some sample 2D data.
    points = np.random.multivariate_normal([0, 0], [[1, 0.5],[0.5, 1]], size=100)
    bandwidth = compute_bandwidth(points)
    print("Estimated bandwidth:", bandwidth)



if __name__ == "__main__":

    # using argparse for argument handling
    parser = argparse.ArgumentParser(description="SING computation of given data.")
    parser.add_argument("--filename", type=str, help="The name of the file to process")
    parser.add_argument("--filetype", type=str, choices=["stipples", "species", "disks"], help="The type of the file (stipples, species, disks)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for SING")
    parser.add_argument("--density", type=float, default=0.0, help="Density value for SING")
    parser.add_argument("--drawEdges", type=bool, default=False, help="Draw SING edges")
    parser.add_argument("--bandwidth", type=float, default=None, help="Bandwidth for KDE")
    args = parser.parse_args()

    filename = args.filename
    filetype = args.filetype
    epsilon = args.epsilon
    density = args.density
    shouldDrawEdges = args.drawEdges
    bandwidth = args.bandwidth

    shouldDraw = True

    # random.seed(1335)

    distance_matrix = []    
    if filetype == "stipples":
        points, distance_matrix, lower_tri = processStippleFile(filename, bandwidth=None, epsilon=epsilon, density = density, shouldDraw = shouldDraw, shouldDrawEdges = shouldDrawEdges)
    elif filetype == "species":
        points, radii, distance_matrix, lower_tri = processDiskFile(filename, epsilon, shouldDrawEdges = shouldDrawEdges)
    elif filetype == "disks":
        points, distance_matrix, lower_tri = processBasicDiskFile(filename, epsilon, shouldDrawEdges = shouldDrawEdges)

    # plt.savefig(filename+str(epsilon)+"_basic.pdf",bbox_inches='tight', pad_inches=0)

    diag = compute_persistence_diagram(distance_matrix)

    barcode = gudhi.plot_persistence_barcode(diag)

    if shouldDraw:
        plt.show()



    