import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as mcolors
import numpy as np

from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

def drawPrincipalAxis(ax, points, ref_idx, k_nn=5, color='yellow', scale=1.0):
    """
    Draws the principal axis (1D PCA direction) for 'ref_idx' and its k_nn neighbors.
    
    Parameters:
      ax       : matplotlib axes to draw on
      points   : Nx2 array of 2D coordinates
      ref_idx  : index of the reference point in 'points'
      k_nn     : number of nearest neighbors to consider
      color    : line color for the axis
      scale    : how long to draw the axis line segments
    """
    # Build a KD-tree for nearest neighbors
    tree = cKDTree(points)
    
    # Query k_nn + 1 because the first neighbor is the point itself
    dists, neighbor_idxs = tree.query(points[ref_idx], k=k_nn+1)
    
    # Exclude the point itself; we only want the neighbors
    neighbor_coords = points[neighbor_idxs[1:]]
    
    # Perform PCA on these neighbors
    pca = PCA(n_components=1)
    pca.fit(neighbor_coords)
    principal = pca.components_[0]  # the main direction (length=1 vector)
    
    # Optionally scale by average neighbor distance for a "nice" line length
    line_len = scale * np.mean(dists[1:])
    
    # Draw the axis in both directions from the reference point
    ref_point = points[ref_idx]
    end_pos = ref_point + line_len * principal
    end_neg = ref_point - line_len * principal
    
    ax.plot([ref_point[0], end_pos[0]], [ref_point[1], end_pos[1]], 
            color=color, linewidth=2)
    ax.plot([ref_point[0], end_neg[0]], [ref_point[1], end_neg[1]], 
            color=color, linewidth=2)


def createPlot(xlim = 0, ylim = 0, Xlim=1000, Ylim=1000):
    
    fig, ax = plt.subplots()
    plt.xlim([xlim,Xlim])
    plt.ylim([ylim,Ylim])
    ax.set_aspect("equal")
    plt.tight_layout()
    ax.axis("off")

    return plt, ax


def createBasicPlot():
    
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    plt.tight_layout()
    ax.axis("off")

    return plt, ax



def drawEdges(ax, edges, points, color='gray', zorder=0):
    
    for edge in edges:
        x_line=[points[edge[0]][0], points[edge[1]][0]]
        y_line=[points[edge[0]][1], points[edge[1]][1]]
        ax.plot(x_line, y_line, color=color, linewidth = 1, zorder = zorder)


def drawCirclesPerClassNoSpecies(data, points, no_classes, class_per_point, ax, offset = 1):
    color = cm.rainbow(np.linspace(0, 1, no_classes))
    
    for index, el in enumerate(data):
        circle = plt.Circle((el[offset], el[offset + 1]), el[offset + 2], zorder=el[offset + 2], color=color[class_per_point[index]], fill=False, alpha = 0.5)

        ax.add_patch(circle)

    min_x, max_x, min_y, max_y = get_data_bounds(points)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

pointsize = 2

def get_data_bounds( points):
    all_x = []
    all_y = []
    for point in points:
        x, y = point
        all_x.append(x)
        all_y.append(y)
    return min(all_x)-pointsize, max(all_x)+pointsize, min(all_y)-pointsize, max(all_y)+pointsize


def drawPoints(plt, points, n_components, labels, size=0.7, ax=None):
    color = cm.rainbow(np.linspace(0, 1, n_components))
    np.random.shuffle(color)
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    color_labels = []
    
    for index, point in enumerate(points):
        if(n_components==1):
            index=0
        color_labels.append(color[(int)(labels[index])])
        
    plt.scatter(x, y, s=size, color=color_labels)
    
    min_x, max_x, min_y, max_y = get_data_bounds(points)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

def plotReferenceDistancesWithAxis(points, dist_mat, ref_idx=0, k_nn=5, title="Distance from Reference Point"):
    """
    Plots all points, colored by their metric distance from the reference point,
    and overlays the principal axis (in red) computed from the reference point's k_nn neighbors.
    """
    # Distance from the reference point to every other point.
    dists = dist_mat[ref_idx, :]

    # Set up colormap and normalization.
    norm = mcolors.Normalize(vmin=dists.min(), vmax=dists.max())
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(6,6))

    # Instead of drawing connecting lines, scatter all points colored by distance.
    sc = ax.scatter(points[:,0], points[:,1], c=dists, cmap=cmap, s=2)
    
    # Highlight the reference point.
    rx, ry = points[ref_idx]
    ax.scatter(rx, ry, c='red', marker='*', s=3, label="Reference Point")

    # Add a colorbar.
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Distance from Reference Point")

    # Draw the principal axis in red with an extended scale (scale=2.0).
    # drawPrincipalAxis(ax, points, ref_idx, k_nn=k_nn, color='red', scale=2.0)

    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
