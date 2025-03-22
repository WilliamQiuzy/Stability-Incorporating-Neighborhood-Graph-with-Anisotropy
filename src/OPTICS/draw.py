import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

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

# def drawPoints(plt, points, n_components, labels, size=0.7, ax=None):
#     color = cm.rainbow(np.linspace(0, 1, n_components))
#     np.random.shuffle(color)
#     x = [point[0] for point in points]
#     y = [point[1] for point in points]
#     color_labels = []
    
#     for index, point in enumerate(points):
#         # If the label is -1, assign a fixed noise color (e.g., gray).
#         if labels[index] == -1:
#             color_labels.append('gray')
#         else:
#             # Otherwise, use the color corresponding to the cluster label.
#             # (Make sure the label is within the proper range.)
#             color_labels.append(color[(int)(labels[index])])
        
#     plt.scatter(x, y, s=size, color=color_labels)
    
#     min_x, max_x, min_y, max_y = get_data_bounds(points)
#     ax.set_aspect('equal', 'box')
#     ax.set_xlim(min_x, max_x)
#     ax.set_ylim(min_y, max_y)
