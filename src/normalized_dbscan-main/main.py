#!/usr/bin/env python3
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial import cKDTree
import gudhi

from compare import *
from input import *
from draw import *
from utils import *

# --------------- Helper Functions for Distance Computations ---------------

def distF(point1, point2, r1, r2):
    """
    Placeholder for the circle metric distance function used in SING.
    Modify this function according to your actual distance definition.
    """
    # For example, here we simply return the Euclidean distance.
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Core distance computation using cKDTree (same as in your framework)
def core_dist(points, k):
    tree = cKDTree(points)
    dists = []
    for point in points:
        ds, inds = tree.query(point, k)
        dists.append(ds[-1])
    return dists

# Compute Mutual Reachability Distances
def computeMRDistances(points, k, filename="", write=False, density=0.0):
    tree = cKDTree(points)
    lower_tri = []
    n = len(points)
    dist_mat = np.zeros((n, n))

    core_dists = core_dist(points, k)

    for i, point in enumerate(points):
        dists = []
        for j in range(0, i):
            distance = max(core_dists[i], core_dists[j], np.linalg.norm(np.array(point) - np.array(points[j])))
            dist_mat[i][j] = distance
            dist_mat[j][i] = distance
            dists.append(distance)
        lower_tri.append(dists)

    if write:
        np.savetxt(filename, dist_mat, delimiter=",")

    return dist_mat, lower_tri

# Compute SING circle distances (for species/disks)
def computeSINGCircleDistances(points, radii, write=False, filename=""):
    n = len(points)
    nn_dists = np.full(n, np.inf)
    dist_mat = np.zeros((n, n))

    # find nearest neighbor for each point using the circle metric
    for i, point in enumerate(points):
        for j in range(0, i):
            d = distF(point, points[j], radii[i], radii[j])
            if d < nn_dists[i]:
                nn_dists[i] = d
            if d < nn_dists[j]:
                nn_dists[j] = d

    # compute the lower triangular matrix needed for persistence computation
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

# Extract SING edges from a distance matrix (for drawing / connectivity)
def extractSINGEdges(dist_mat, epsilon=1.0):
    n = len(dist_mat)
    edges = []
    adj_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(0, i):
            if dist_mat[i][j] <= epsilon:
                edges.append((i, j))
                adj_mat[i][j] = 1
                adj_mat[j][i] = 1
    return edges, adj_mat

# ------------------- File Processing Functions -------------------

def processDiskFile(filename, epsilon=1.0, shouldDraw=True, shouldDrawEdges=False):
    """
    Process a disk (or species) file.
    The file is assumed to contain extra information; the function returns:
      data, species, xs, ys, radii, colours, points, species_labels
    and computes SING circle distances.
    """
    data, species, xs, ys, radii, colours, points, species_labels = readDiskFile(filename)
    dist_mat, lower_tri = computeSINGCircleDistances(points, radii)
    
    if shouldDraw:
        plt.figure()
        ax = plt.gca()
        edges, adj_mat = extractSINGEdges(dist_mat, epsilon)
        if shouldDrawEdges:
            drawEdges(ax, edges, points)
        n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)
        print(f"Number of components: {n_components}")
        drawCirclesPerClassNoSpecies(data, points, n_components, labels, ax)
        plt.title("Disk File Clustering")
        plt.show()
    
    return points, radii, dist_mat, lower_tri

def processBasicDiskFile(filename, epsilon=1.0, shouldDraw=True, shouldDrawEdges=False):
    """
    Process a basic disk file (with only coordinates and radii).
    Returns points, dist_mat, lower_tri.
    """
    data, xs, ys, radii, points = readBasicDiskFile(filename)
    dist_mat, lower_tri = computeSINGCircleDistances(points, radii)
    
    if shouldDraw:
        plt.figure()
        ax = plt.gca()
        ax.set_aspect("equal")
        edges, adj_mat = extractSINGEdges(dist_mat, epsilon)
        if shouldDrawEdges:
            drawEdges(ax, edges, points)
        n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)
        print(f"Number of components: {n_components}")
        drawCirclesPerClassNoSpecies(data, points, n_components, labels, ax, 0)
        plt.title("Basic Disk File Clustering")
        plt.show()
    
    return points, dist_mat, lower_tri

def processStippleFile(filename, k=5, epsilon=1.0, density=0.0, shouldDraw=False, shouldDrawEdges=False):
    """
    Process a stipple file (that only contains 2D coordinates).
    Returns points, dist_mat, lower_tri.
    """
    points, xs, ys = readStipplefile(filename)
    dist_mat, lower_tri = computeMRDistances(points, k=k, filename=filename+"_distmat.txt", write=False, density=density)
    
    if shouldDraw:
        plt.figure()
        ax = plt.gca()
        edges, adj_mat = extractSINGEdges(dist_mat, epsilon)
        if shouldDrawEdges:
            drawEdges(ax, edges, points)
        n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)
        print(f"Number of components: {n_components}")
        drawPoints(plt, points, n_components, labels, 2, ax)
        plt.title("Stipple File Clustering")
        plt.show()
    
    return points, dist_mat, lower_tri

# ------------------- Simplified HDBSCAN Clustering -------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        # In a full HDBSCAN implementation, birth/death lambda values are tracked.
        self.birth_lambda = [np.inf] * n
        self.death_lambda = [0.0] * n

    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j, lambda_val):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return
        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
            self.size[root_j] += self.size[root_i]
            self.death_lambda[root_i] = lambda_val
        else:
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.death_lambda[root_j] = lambda_val
            if self.rank[root_i] == self.rank[root_j]:
                self.rank[root_i] += 1

class HDBSCAN:
    def __init__(self, min_cluster_size=5):
        """
        Parameters:
          min_cluster_size: minimum number of points required to form a cluster.
        """
        self.min_cluster_size = min_cluster_size
        self.labels_ = None
        self.dist_mat = None
        self.lower_tri = None
        self.points = None

    def fit(self, points, dist_mat=None, lower_tri=None, k=None, use_mrdist=True):
        """
        Fit HDBSCAN on the given points. Optionally a precomputed distance matrix
        and lower-triangular matrix can be provided.
        
        If not provided, the code computes them using:
          - computeMRDistances (for stipple files) if use_mrdist is True, or
          - (by default) computeMRDistances.
        """
        self.points = np.asarray(points)
        n_points = self.points.shape[0]
        if dist_mat is None or lower_tri is None:
            if k is None:
                k = self.min_cluster_size
            if use_mrdist:
                print("Computing mutual reachability distances using MRDistances...")
                dist_mat, lower_tri = computeMRDistances(self.points, k=k)
            else:
                # For species/disks the circle metric might be used.
                print("Computing mutual reachability distances (default) using MRDistances...")
                dist_mat, lower_tri = computeMRDistances(self.points, k=k)
        self.dist_mat = dist_mat
        self.lower_tri = lower_tri

        # Compute the Minimum Spanning Tree (MST) over the mutual reachability graph.
        print("Computing Minimum Spanning Tree (MST)...")
        mst_sparse = minimum_spanning_tree(dist_mat)
        mst_matrix = mst_sparse.toarray()

        # Extract edges from the MST.
        edges = []
        for i in range(n_points):
            for j in range(n_points):
                if mst_matrix[i, j] > 0:
                    edges.append((i, j, mst_matrix[i, j]))
        edges.sort(key=lambda edge: edge[2])

        # Build a simplified cluster hierarchy using union-find.
        print("Building cluster hierarchy...")
        uf = UnionFind(n_points)
        dendrogram = []  # For a full implementation, store the dendrogram.
        for i, j, weight in edges:
            if uf.find(i) != uf.find(j):
                lambda_val = 1.0 / (weight + 1e-10)
                uf.union(i, j, lambda_val)
                dendrogram.append((i, j, weight, lambda_val))

        # Extract clusters (flat clustering) from the union-find structure.
        print("Extracting clusters...")
        clusters = {}
        for i in range(n_points):
            root = uf.find(i)
            clusters.setdefault(root, []).append(i)
        labels = -np.ones(n_points, dtype=int)
        cluster_id = 0
        for cluster_points in clusters.values():
            if len(cluster_points) >= self.min_cluster_size:
                for point in cluster_points:
                    labels[point] = cluster_id
                cluster_id += 1

        self.labels_ = labels
        print("Clustering complete. Number of clusters found:", cluster_id)
        return self

# ------------------- Main Routine with Argument Parsing -------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDBSCAN clustering with custom input/output.")
    parser.add_argument("--filename", type=str, required=True, help="The name of the file to process")
    parser.add_argument("--filetype", type=str, choices=["stipples", "species", "disks"], required=True,
                        help="The type of the file (stipples, species, disks)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value for SING edge extraction")
    parser.add_argument("--density", type=float, default=0.0, help="Density value for SING")
    parser.add_argument("--drawEdges", type=bool, default=False, help="Draw SING edges")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors for core distance computation")
    parser.add_argument("--min_cluster_size", type=int, default=10, help="Minimum cluster size for HDBSCAN")
    args = parser.parse_args()

    filename = args.filename
    filetype = args.filetype
    epsilon = args.epsilon
    density = args.density
    drawEdges = args.drawEdges
    k = args.k
    min_cluster_size = args.min_cluster_size

    random.seed(1335)
    shouldDraw = True

    # Process the input file based on its type.
    if filetype == "stipples":
        points, dist_mat, lower_tri = processStippleFile(filename, k=k, epsilon=epsilon,
                                                         density=density, shouldDraw=shouldDraw,
                                                         shouldDrawEdges=drawEdges)
    elif filetype == "species":
        points, radii, dist_mat, lower_tri = processDiskFile(filename, epsilon=epsilon,
                                                             shouldDraw=shouldDraw,
                                                             shouldDrawEdges=drawEdges)
    elif filetype == "disks":
        points, dist_mat, lower_tri = processBasicDiskFile(filename, epsilon=epsilon,
                                                           shouldDraw=shouldDraw,
                                                           shouldDrawEdges=drawEdges)
    else:
        raise ValueError("Unsupported file type.")

    # Run HDBSCAN clustering using the precomputed distance matrix and lower-triangular matrix.
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(points, dist_mat=dist_mat, lower_tri=lower_tri, k=k)
    labels = clusterer.labels_
    print("Cluster labels:", labels)

    # Compute the persistence diagram (using a function from your utils/compare modules)
    diag = compute_persistence_diagram(dist_mat)
    gudhi.plot_persistence_barcode(diag)

    # Optionally draw the clustering result.
    if shouldDraw:
        plt.figure()
        if filetype == "stipples":
            # Assuming drawPoints is defined in your draw module.
            drawPoints(plt, points, len(np.unique(labels)), labels, 2, plt.gca())
        elif filetype in ["species", "disks"]:
            # For disk files we can simply plot the points colored by cluster.
            points_arr = np.array(points)
            plt.scatter(points_arr[:, 0], points_arr[:, 1], c=labels, cmap="viridis", s=50)
        plt.title("HDBSCAN Clustering")
        plt.show()
