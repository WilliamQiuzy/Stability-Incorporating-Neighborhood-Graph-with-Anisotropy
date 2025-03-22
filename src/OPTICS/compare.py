import numpy as np

import pickle as pickle
from pylab import *
import gudhi


######### compute persistence diagram for original data

def compute_persistence_diagram(distance_matrix):
    rips_complex = gudhi.RipsComplex(
    distance_matrix=distance_matrix, max_edge_length=np.inf
    )
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)

    diag = simplex_tree.persistence()

    return diag


