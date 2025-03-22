import numpy as np

import pickle as pickle
from pylab import *
import gudhi


######### compute persistence diagram for original data

def compute_persistence_diagram(distance_matrix, max_edge=1.0):
    """
    Build a Rips complex up to max_edge, then compute the persistence diagram.
    """
    # Limit the filtration to [0, max_edge].
    rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge)
    st = rips_complex.create_simplex_tree(max_dimension=2)
    diag = st.persistence()
    return diag

