#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import heapq
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import argparse

# ================================
# Modified OPTICS Algorithm (normalized metric)
# ================================
def optics(X, eps, min_pts):
    """
    Run OPTICS on data X using a normalized distance metric.
    
    For two points i and j the normalized distance is defined as:
    
        d_norm(i,j) = ||X[i]-X[j]|| / (nn[i] + nn[j] + 1e-10)
    
    where nn[i] is the distance from point i to its nearest neighbor (excluding itself).
    
    Parameters:
      X       : array-like, shape (n_points, n_features)
      eps     : float, radius for neighbor search (in original space)
      min_pts : int, minimum number of points (including self) to be considered dense.
      
    Returns:
      ordering        : list of int, the order in which points are processed.
      reachability    : ndarray, reachability distance for each point.
      core_distances  : ndarray, core distance for each point.
    """
    n_points = X.shape[0]
    reachability = np.full(n_points, np.inf)
    processed = np.zeros(n_points, dtype=bool)
    ordering = []
    core_distances = np.full(n_points, np.nan)

    # Build a KD-tree using the original Euclidean metric.
    tree = cKDTree(X)
    # Precompute nearest neighbor distances (k=2: self and the closest other point)
    dists_nn, _ = tree.query(X, k=2)
    nn = dists_nn[:, 1]  # nearest neighbor distance for each point

    def get_neighbors(idx):
        """Return indices of all points within eps of X[idx] (self is included)."""
        return tree.query_ball_point(X[idx], eps)

    def normalized_distance(i, j):
        d = np.linalg.norm(X[i] - X[j])
        return d
        # return d

    def update(idx, neighbors, seeds):
        for o in neighbors:
            if not processed[o]:
                d_norm = normalized_distance(idx, o)
                new_reach = max(core_distances[idx], d_norm)
                if reachability[o] == np.inf:
                    reachability[o] = new_reach
                    heapq.heappush(seeds, (reachability[o], o))
                else:
                    if new_reach < reachability[o]:
                        reachability[o] = new_reach
                        heapq.heappush(seeds, (reachability[o], o))

    # Main loop.
    for i in range(n_points):
        if not processed[i]:
            neighbors = get_neighbors(i)
            processed[i] = True
            ordering.append(i)
            if len(neighbors) >= min_pts:
                dists = [normalized_distance(i, j) for j in neighbors]
                dists.sort()
                if len(dists) >= min_pts:
                    core_distances[i] = dists[min_pts - 1]
                else:
                    core_distances[i] = np.nan
                seeds = []
                update(i, neighbors, seeds)
                while seeds:
                    current_reach, j = heapq.heappop(seeds)
                    if processed[j]:
                        continue
                    neighbors_j = get_neighbors(j)
                    processed[j] = True
                    ordering.append(j)
                    if len(neighbors_j) >= min_pts:
                        dists_j = [normalized_distance(j, k) for k in neighbors_j]
                        dists_j.sort()
                        if len(dists_j) >= min_pts:
                            core_distances[j] = dists_j[min_pts - 1]
                        else:
                            core_distances[j] = np.nan
                        update(j, neighbors_j, seeds)
    return ordering, reachability, core_distances

# ================================
# Xi-based and Manually-based Extraction (previous method)
# ================================
def extract_clusters(ordering, reachability, core_distances, threshold):
    """
    Extract clusters from the OPTICS ordering using a simple DBSCAN-like procedure.
    A new cluster is started when a point's reachability exceeds the threshold.

    Parameters:
        ordering      : list of int
                        The OPTICS ordering (indices of points in the order they were processed).
        reachability  : ndarray
                        The reachability distances for each point.
        core_distances: ndarray
                        The core distances for each point.
        threshold     : float
                        The reachability threshold for starting a new cluster.
                        
    Returns:
        labels        : ndarray, shape (n_points,)
                        Cluster labels for each point. Noise points are labeled as -1.
    """
    n_points = len(reachability)
    labels = -np.ones(n_points, dtype=int)
    cluster_id = 0

    for i, idx in enumerate(ordering):
        if reachability[idx] > threshold:
            if not np.isnan(core_distances[idx]):
                labels[idx] = cluster_id
                cluster_id += 1
            else:
                labels[idx] = -1
        else:
            if i == 0:
                if not np.isnan(core_distances[idx]):
                    labels[idx] = cluster_id
                    cluster_id += 1
                else:
                    labels[idx] = -1
            else:
                prev_idx = ordering[i-1]
                if labels[prev_idx] != -1:
                    labels[idx] = labels[prev_idx]
                else:
                    if not np.isnan(core_distances[idx]):
                        labels[idx] = cluster_id
                        cluster_id += 1
                    else:
                        labels[idx] = -1
    return labels

def extract_xi_clusters(ordering, reachability, xi):
    r_order = [reachability[i] for i in ordering]
    n = len(r_order)
    intervals = []
    start = None
    for i in range(1, n):
        if (np.isfinite(r_order[i-1]) and np.isfinite(r_order[i]) and 
            r_order[i-1] > 0 and (r_order[i-1] - r_order[i]) / r_order[i-1] >= xi):
            if start is None:
                start = i - 1
        if start is not None and i < n:
            if (np.isfinite(r_order[i]) and np.isfinite(r_order[i-1]) and 
                r_order[i] > 0 and (r_order[i] - r_order[i-1]) / r_order[i] >= xi):
                end = i
                intervals.append((start, end))
                start = None
    labels_order = -np.ones(n, dtype=int)
    cluster_id = 0
    for (s, e) in intervals:
        for i in range(s, e + 1):
            labels_order[i] = cluster_id
        cluster_id += 1
    global_labels = -np.ones(len(ordering), dtype=int)
    for order_index, global_index in enumerate(ordering):
        global_labels[global_index] = labels_order[order_index]
    return global_labels

# ================================
# Automatic Extraction (from Sander et al. 2003)
# ================================
import sys  # For exception handling

def isLocalMaxima(index, RPlot, RPoints, nghsize):
    for i in range(1, nghsize+1):
        if index + i < len(RPlot):
            if RPlot[index] < RPlot[index+i]:
                return 0
        if index - i >= 0:
            if RPlot[index] < RPlot[index-i]:
                return 0
    return 1

def findLocalMaxima(RPlot, RPoints, nghsize):
    localMaximaPoints = {}
    for i in range(1, len(RPoints)-1):
        if RPlot[i] > RPlot[i-1] and RPlot[i] >= RPlot[i+1] and isLocalMaxima(i, RPlot, RPoints, nghsize) == 1:
            localMaximaPoints[i] = RPlot[i]
    return sorted(localMaximaPoints, key=localMaximaPoints.__getitem__, reverse=True)

class TreeNode(object):
    def __init__(self, points, start, end, parentNode):
        self.points = points
        self.start = start
        self.end = end
        self.parentNode = parentNode
        self.children = []
        self.splitpoint = -1
    def __str__(self):
        return "start: %d, end: %d, split: %d" % (self.start, self.end, self.splitpoint)
    def assignSplitPoint(self, splitpoint):
        self.splitpoint = splitpoint
    def addChild(self, child):
        self.children.append(child)

def clusterTree(node, parentNode, localMaximaPoints, RPlot, RPoints, min_cluster_size):
    if len(localMaximaPoints) == 0:
        return
    s = localMaximaPoints[0]
    node.assignSplitPoint(s)
    localMaximaPoints = localMaximaPoints[1:]
    Node1 = TreeNode(RPoints, node.start, s, node)
    Node2 = TreeNode(RPoints, s+1, node.end, node)
    LocalMax1 = []
    LocalMax2 = []
    for i in localMaximaPoints:
        if i < s:
            LocalMax1.append(i)
        if i > s:
            LocalMax2.append(i)
    Nodelist = [(Node1, LocalMax1), (Node2, LocalMax2)]
    significantMin = 0.003
    if RPlot[s] < significantMin:
        node.assignSplitPoint(-1)
        clusterTree(node, parentNode, localMaximaPoints, RPlot, RPoints, min_cluster_size)
        return
    checkRatio = 0.8
    checkValue1 = int(round(checkRatio * len(Node1.points)))
    checkValue2 = int(round(checkRatio * len(Node2.points)))
    if checkValue2 == 0:
        checkValue2 = 1
    avgReachValue1 = float(np.average(RPlot[node.end - checkValue1: node.end]))
    avgReachValue2 = float(np.average(RPlot[Node2.start: Node2.start+checkValue2]))
    maximaRatio = 0.75
    rejectionRatio = 0.7
    if float(avgReachValue1 / float(RPlot[s])) > maximaRatio or float(avgReachValue2 / float(RPlot[s])) > maximaRatio:
        if float(avgReachValue1 / float(RPlot[s])) < rejectionRatio:
            try:
                Nodelist.remove((Node2, LocalMax2))
            except Exception:
                pass
        if float(avgReachValue2 / float(RPlot[s])) < rejectionRatio:
            try:
                Nodelist.remove((Node1, LocalMax1))
            except Exception:
                pass
        if float(avgReachValue1 / float(RPlot[s])) >= rejectionRatio and float(avgReachValue2 / float(RPlot[s])) >= rejectionRatio:
            node.assignSplitPoint(-1)
            clusterTree(node, parentNode, localMaximaPoints, RPlot, RPoints, min_cluster_size)
            return
    if len(Node1.points) < min_cluster_size:
        try:
            Nodelist.remove((Node1, LocalMax1))
        except Exception:
            pass
    if len(Node2.points) < min_cluster_size:
        try:
            Nodelist.remove((Node2, LocalMax2))
        except Exception:
            pass
    if len(Nodelist) == 0:
        node.assignSplitPoint(-1)
        return
    similaritythreshold = 0.4
    bypassNode = 0
    if parentNode is not None:
        sumRP = np.average(RPlot[node.start:node.end])
        sumParent = np.average(RPlot[parentNode.start:parentNode.end])
        if float((node.end - node.start) / float(parentNode.end - parentNode.start)) > similaritythreshold:
            parentNode.children.remove(node)
            bypassNode = 1
    for (childNode, localMaxList) in Nodelist:
        if bypassNode == 1:
            parentNode.addChild(childNode)
            clusterTree(childNode, parentNode, localMaxList, RPlot, RPoints, min_cluster_size)
        else:
            node.addChild(childNode)
            clusterTree(childNode, node, localMaxList, RPlot, RPoints, min_cluster_size)

def automaticCluster(RPlot, RPoints):
    min_cluster_size_ratio = 0.005
    min_neighborhood_size = 2
    min_maxima_ratio = 0.001
    min_cluster_size = int(min_cluster_size_ratio * len(RPoints))
    if min_cluster_size < 5:
        min_cluster_size = 5
    nghsize = int(min_maxima_ratio * len(RPoints))
    if nghsize < min_neighborhood_size:
        nghsize = min_neighborhood_size
    localMaximaPoints = findLocalMaxima(RPlot, RPoints, nghsize)
    rootNode = TreeNode(RPoints, 0, len(RPoints), None)
    clusterTree(rootNode, None, localMaximaPoints, RPlot, RPoints, min_cluster_size)
    return rootNode

def getLeaves(node, arr):
    if node is not None:
        if node.splitpoint == -1:
            arr.append(node)
        for n in node.children:
            getLeaves(n, arr)
    return arr

def extract_clusters_auto(ordering, reachability):
    """
    Given the OPTICS ordering and reachability array, construct a reachability plot (RPlot)
    and use the automatic clustering method to build a hierarchical cluster tree.
    Then extract clusters (leaf nodes) and assign a unique label to each leaf's region.
    
    Returns an array of global labels (of length equal to the number of data points).
    """
    RPlot = [reachability[i] for i in ordering]
    RPoints = ordering[:]  # our ordered list of indices
    root = automaticCluster(RPlot, RPoints)
    leaves = getLeaves(root, [])
    labels = -np.ones(len(ordering), dtype=int)
    cluster_id = 0
    for leaf in leaves:
        for i in range(leaf.start, leaf.end):
            labels[RPoints[i]] = cluster_id
        cluster_id += 1
    return labels

def drawOPTICSEdges(ax, X, eps, edge_threshold=0.5, color='gray', labels=None):
    """
    Draws edges for OPTICS: for each point in X, find all neighbors within eps (Euclidean),
    compute the normalized distance (using the same nearest-neighbor values as in the OPTICS
    calculation), and draw an edge if:
      1. The normalized distance is below edge_threshold, and
      2. (If provided) both points belong to the same cluster (labels != -1 and equal).
    
    Parameters:
      ax            : matplotlib axes to draw on.
      X             : (n_points, n_features) numpy array of data points.
      eps           : float, the radius used for neighbor search.
      edge_threshold: float, only draw an edge if normalized_distance < edge_threshold.
      color         : color for the edges.
      labels        : (optional) array of cluster labels; if provided, only draw edges
                      between points with the same label (and non-noise).
    """
    # Build KD-tree and compute each point's nearest neighbor distance (excluding self)
    tree = cKDTree(X)
    dists_nn, _ = tree.query(X, k=2)
    nn = dists_nn[:, 1]  # nearest neighbor distance for each point

    n_points = X.shape[0]
    for i in range(n_points):
        neighbors = tree.query_ball_point(X[i], r=eps)
        for j in neighbors:
            if j <= i:
                continue
            # If cluster labels provided, only connect points in the same cluster (and not noise)
            if labels is not None:
                if labels[i] == -1 or labels[j] == -1:
                    continue
                if labels[i] != labels[j]:
                    continue
            d = np.linalg.norm(X[i] - X[j])
            norm_d = d / (nn[i] + nn[j] + 1e-10)
            if norm_d < edge_threshold:
                ax.plot([X[i, 0], X[j, 0]],
                        [X[i, 1], X[j, 1]],
                        color=color, linewidth=0.5, alpha=0.5)


def plotOPTICSEdges(X, eps, edge_threshold=0.5, labels=None):
    """
    Visualize OPTICS data by plotting points and overlaying edges computed with drawOPTICSEdges.
    If 'labels' are provided (from the OPTICS clustering extraction), points are colored by cluster,
    and edges are drawn only between points in the same cluster.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    if labels is not None:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap("viridis", len(unique_labels))
        for idx, lab in enumerate(unique_labels):
            pts = X[labels == lab]
            if lab == -1:
                ax.scatter(pts[:, 0], pts[:, 1], s=2, c='gray', zorder=2, label="Noise")
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=2, c=[cmap(idx)], zorder=2, label=f"Cluster {lab}")
    else:
        ax.scatter(X[:, 0], X[:, 1], s=2, c='blue', zorder=2)
    
    drawOPTICSEdges(ax, X, eps, edge_threshold=edge_threshold, color='gray', labels=labels)
    ax.set_aspect("equal")
    ax.set_title("OPTICS: Edges (edge_threshold={})".format(edge_threshold))
    ax.legend(fontsize=8)
    plt.show()

# ================================
# Main Function
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OPTICS clustering with automatic extraction of clusters from hierarchical representations."
    )
    parser.add_argument("--filename", type=str, required=True, help="Input filename")
    parser.add_argument("--filetype", type=str, choices=["stipples", "species", "disks"], required=True, help="Input file type")
    parser.add_argument("--eps", type=float, default=np.inf, help="eps for neighbor search (in original space)")
    parser.add_argument("--min_pts", type=int, default=5, help="Minimum points to form a dense region")
    parser.add_argument("--extraction", type=str, choices=["xi", "auto", "manual"], default="manual",
                        help="Cluster extraction method: 'xi' for xi-based, 'auto' for the automatic method, 'manual' for manual")
    parser.add_argument("--xi", type=float, default=0.05, help="Xi threshold for xi-based extraction (if used)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Reachability threshold for manual extraction (if used)")
    args = parser.parse_args()

    # Import custom input functions.
    from input import readStipplefile, readDiskFile, readBasicDiskFile
    # Import drawing functions.
    from draw import createBasicPlot, drawPoints

    # Read the dataset.
    if args.filetype == "stipples":
        points, xs, ys = readStipplefile(args.filename)
    elif args.filetype == "species":
        data, species, xs, ys, radii, colours, points, species_labels = readDiskFile(args.filename)
    elif args.filetype == "disks":
        data, xs, ys, radii, points = readBasicDiskFile(args.filename)
    else:
        raise ValueError("Unsupported file type provided.")

    X = np.array(points)
    print("Input length:", len(X))

    # Normalize data (min-max scaling).
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    print("Data normalized using min-max scaling.")

    # Run OPTICS.
    ordering, reachability, core_distances = optics(X_norm, args.eps, args.min_pts)
    # from sklearn.cluster import OPTICS

    # optics_model = OPTICS(min_samples=args.min_pts,
    #                     max_eps=args.eps,
    #                     cluster_method='xi',
    #                     xi=args.xi,
    #                     metric='euclidean')
    # optics_model.fit(X_norm)

    # ordering = optics_model.ordering_
    # reachability = optics_model.reachability_
    # core_distances = optics_model.core_distances_
    # labels = optics_model.labels_

    # Plot the reachability plot.
    reachability_ordered = [reachability[i] for i in ordering]
    plt.figure(figsize=(10,7))
    plt.bar(range(len(ordering)), reachability_ordered, color='b')
    plt.xlabel("Ordering Index")
    plt.ylabel("Normalized Reachability Distance")
    plt.show()

    # Cluster extraction.
    if args.extraction == "xi":
        labels = extract_xi_clusters(ordering, reachability, args.xi)
    elif args.extraction == "manual":
        labels = extract_clusters(ordering, reachability, core_distances, args.threshold)
    else:
        labels = extract_clusters_auto(ordering, reachability)

    # Visualize the clusters.
    plt_obj, ax = createBasicPlot()
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    # Use smaller dot size.
    drawPoints(plt_obj, points, n_clusters, labels, size=2, ax=ax)
    plt_obj.show()

    # Suppose X is your (n_points,2) data and eps is your neighbor radius.
    plotOPTICSEdges(X, eps=args.eps, edge_threshold=args.threshold)




