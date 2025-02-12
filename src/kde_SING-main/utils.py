import numpy as np
from collections import Counter
from draw import *
from sklearn.cluster import DBSCAN

def euclidean_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def distF(a, b, r_a, r_b):
    if(r_a < r_b):
        temp = r_a
        r_a = r_b
        r_b = temp

        temp_center = a
        a = b
        b = temp_center


    d = np.linalg.norm(a - b)
    
    extent = max(d + r_a + r_b, 2*r_a)
    overlap = np.clip(r_a + r_b - d, 0, 2*r_b)
    f = extent - overlap + d + r_a - r_b

    if d <= r_a - r_b:
        return f/(4*r_a - 4*r_b)
    elif r_a - r_b < d and d <= r_a + r_b:
        return (f - 4*r_a + 7*r_b)/(3* r_b)
    else:
        return f - 4*r_a + 2*r_b + 3
    

def DBScanCluster(points, eps):
    
    clustering = DBSCAN(eps).fit(points)
    
    plt, ax = createBasicPlot()

    print(f"Number of DBSCAN components: {len(Counter(clustering.labels_).keys())}")

    drawPoints(plt, points, len(Counter(clustering.labels_).keys()), clustering.labels_, size=2, ax=ax)
    return len(Counter(clustering.labels_).keys()), clustering.labels_