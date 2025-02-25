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


def drawPoints(plt_obj, points, n_components, labels, size=0.7, ax=None):
    from matplotlib import cm
    # If there are no clusters, assign a default color array.
    if n_components <= 0:
        color = np.array(['gray'])
    else:
        color = cm.rainbow(np.linspace(0, 1, n_components))
        np.random.shuffle(color)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    color_labels = []
    
    for idx, label in enumerate(labels):
        if label == -1:
            color_labels.append('gray')
        else:
            # Use modulo in case label is outside the expected range.
            color_labels.append(color[int(label) % len(color)])
        
    ax.scatter(xs, ys, s=size, color=color_labels)
    
    # Assuming get_data_bounds is defined elsewhere:
    min_x, max_x, min_y, max_y = get_data_bounds(points)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)


