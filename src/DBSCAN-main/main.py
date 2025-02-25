#!/usr/bin/env python3
"""
DBSCAN Clustering for Stipple Data, with existing SING code for other file types.

In this version, we replace the SING/KDE pipeline for stipple files with a
DBSCAN clustering (using scikit-learn). For species/disks, we keep the old code.

Usage example:
    python main.py --filename='./examples/stipples/Tree.csv' --filetype='stipples' --epsilon=1.0 --min_samples=5
"""

import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from scipy.sparse.csgraph import connected_components
import gudhi

# Imports from your existing modules
from compare import *
from input import *
from draw import *
from utils import *

#############################
# (Unused) KDE-based function (kept for reference)
#############################

def computeKDEProximityDistances(points, bandwidth=None, write=False, filename=""):
    """
    (No longer used for stipple files, but kept here for reference.)
    """
    # This function is no longer called for stipple data.
    # We'll leave it in place so that the rest of the code doesn't break.
    pass

#############################
# Circle Metric SING (unchanged for species/disks)
#############################

def computeSINGCircleDistances(points, radii, write=False, filename=""):
    nn_dists = np.full(len(points), np.inf)
    dist_mat = np.zeros((len(points), len(points)))
    for i, point in enumerate(points):
        for j in range(0, i):
            dist = distF(point, points[j], radii[i], radii[j])
            if dist < nn_dists[i]:
                nn_dists[i] = dist
            if dist < nn_dists[j]:
                nn_dists[j] = dist
    lower_tri = []
    for i, point in enumerate(points):
        dists = []
        for j in range(0, i):
            distance = distF(point, points[j], radii[i], radii[j]) / (nn_dists[i] + nn_dists[j])
            dist_mat[i][j] = distance
            dist_mat[j][i] = distance
            dists.append(distance)
        lower_tri.append(dists)
    if write:
        np.savetxt(filename, dist_mat, delimiter=",")
    return dist_mat, lower_tri

#############################
# SING Edge Extraction (unused for DBSCAN but kept for other file types)
#############################

def extractSINGEdges(dist_mat, epsilon=1.0):
    n = len(dist_mat[0])
    edges = []
    adj_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if dist_mat[i][j] <= epsilon:
                edges.append((i, j))
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1
    return edges, adj_mat

#############################
# Disk/Species File Processing (unchanged)
#############################

def processDiskFile(filename, epsilon=1.0, shouldDraw=True, shouldDrawEdges=False):
    data, species, xs, ys, radii, colours, points, species_labels = readDiskFile(filename)
    dist_mat, lower_tri = computeSINGCircleDistances(points, radii)
    if shouldDraw:
        plt_obj, ax = createPlot(0, 0, 10000, 10000)
        edges, adj_mat = extractSINGEdges(dist_mat, epsilon)
        if shouldDrawEdges:
            drawEdges(ax, edges, points)
        n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)
        print(f"Number of components: {n_components}")
        drawCirclesPerClassNoSpecies(data, points, n_components, labels, ax)
    return points, radii, dist_mat, lower_tri

def processBasicDiskFile(filename, epsilon=1.0, shouldDraw=True, shouldDrawEdges=False):
    data, xs, ys, radii, points = readBasicDiskFile(filename)
    dist_mat, lower_tri = computeSINGCircleDistances(points, radii)
    if shouldDraw:
        plt_obj, ax = createBasicPlot()
        ax.set_aspect("equal")
        edges, adj_mat = extractSINGEdges(dist_mat, epsilon)
        if shouldDrawEdges:
            drawEdges(ax, edges, points)
        n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)
        print(f"Number of components: {n_components}")
        drawCirclesPerClassNoSpecies(data, points, n_components, labels, ax, 0)
    return points, dist_mat, lower_tri

#############################
# Stipple File Processing with DBSCAN
#############################

def processStippleFile(filename, epsilon=1.0, min_samples=5, shouldDraw=True, shouldDrawEdges=False):
    """
    Process a stipple file (2D points) with DBSCAN from scikit-learn.

    We ignore the old SING approach for stipples and directly run DBSCAN on the raw coordinates.
    Then, for consistency, we return a 'distance_matrix' of zeros (or None) so the rest of the
    pipeline does not break, though the persistence diagram won't be meaningful in this case.
    """
    points, xs, ys = readStipplefile(filename)
    points_arr = np.array(points)

    # Run DBSCAN
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean')
    dbscan.fit(points_arr)
    labels = dbscan.labels_
    # The number of clusters (excluding noise):
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"DBSCAN found {n_clusters} clusters (plus noise).")

    # For consistency, build a "distance_matrix" that won't be used for adjacency,
    # but just to keep the pipeline consistent. We store zeros:
    n = len(points)
    distance_matrix = np.zeros((n, n))
    lower_tri = [[] for _ in range(n)]

    # Visualization
    if shouldDraw:
        plt_obj, ax = createBasicPlot()
        drawPoints(plt_obj, points, n_clusters, labels, 2, ax)
        plt_obj.title("DBSCAN Clustering (Stipple File)")
        plt_obj.show()

    return points, distance_matrix, lower_tri

#############################
# Main
#############################

def main():
    parser = argparse.ArgumentParser(description="Process data with DBSCAN for stipples, or SING for others.")
    parser.add_argument("--filename", type=str, required=True, help="The name of the file to process")
    parser.add_argument("--filetype", type=str, choices=["stipples", "species", "disks"], required=True,
                        help="The type of the file (stipples, species, disks)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for DBSCAN or SING")
    parser.add_argument("--min_samples", type=int, default=5, help="min_samples for DBSCAN (stipple only)")
    parser.add_argument("--drawEdges", type=bool, default=False, help="Draw SING edges (only used for species/disks).")
    args = parser.parse_args()

    filename = args.filename
    filetype = args.filetype
    epsilon = args.epsilon
    min_samples = args.min_samples
    shouldDrawEdges = args.drawEdges

    shouldDraw = True

    if filetype == "stipples":
        # Use DBSCAN for stipple data
        points, distance_matrix, lower_tri = processStippleFile(filename,
                                                                epsilon=epsilon,
                                                                min_samples=min_samples,
                                                                shouldDraw=shouldDraw,
                                                                shouldDrawEdges=shouldDrawEdges)
    elif filetype == "species":
        # Old SING approach
        points, radii, distance_matrix, lower_tri = processDiskFile(filename, epsilon,
                                                                    shouldDrawEdges=shouldDrawEdges)
    elif filetype == "disks":
        # Old SING approach
        points, distance_matrix, lower_tri = processBasicDiskFile(filename, epsilon,
                                                                  shouldDrawEdges=shouldDrawEdges)

    # If you still want to compute a persistence diagram, it won't be meaningful for the DBSCAN approach,
    # but we keep it for consistency with your existing pipeline:
    diag = compute_persistence_diagram(distance_matrix)
    barcode = gudhi.plot_persistence_barcode(diag)
    plt.show()

if __name__ == "__main__":
    main()
