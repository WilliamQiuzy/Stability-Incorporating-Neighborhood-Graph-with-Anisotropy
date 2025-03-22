# Stability Incorporating Neighborhood Graph with Anisotropy

This repository provides a collection of clustering algorithms that extend or complement the Stability-Incorporating Neighborhood Graph (SING) framework, focusing on robust performance under noise, outliers, and anisotropic data distributions. Our primary contribution is **KA-SING**, an algorithm that combines local density estimation (via the kth nearest neighbor) with an angular term to better handle directional variations.

---

## Project Goals

- **Enhance SING**: Build on the original SING method by introducing anisotropy and refining local density normalization.
- **Compare Multiple Approaches**: Provide implementations of various clustering algorithms for benchmarking and analysis.
- **Facilitate Research**: Offer easy-to-run examples and extendable code to encourage experimentation in neighborhood-based clustering.

---

## Repository Structure

- **avgknn-main**: Implements the *average of k nearest neighbors* approach for distance normalization.
- **avgknn_angle-main**: Extends avgknn with an angular term to capture anisotropy in the local neighborhood.
- **DBSCAN-main**: A standard density-based clustering algorithm, adapted for easy comparison.
- **K-SING-main**: Uses the kth nearest neighbor distance for local scale estimation, building on the SING framework.
- **KA-SING_angle-main**: Our main contribution, combining kth nearest neighbor and an anisotropic term to capture directional variations.
- **OPTICS**: Another density-based method, offering multi-scale reachability analysis.
- **SING-main**: A direct reference implementation of SING, cloned or adapted from [di-marin/SING](https://github.com/di-marin/SING).

---

## Algorithms Overview

1. **KA-SING**  
   - *Our Method with Angle*: Incorporates both kth nearest neighbor density normalization and an angular term to handle anisotropic datasets.
2. **K-SING**  
   - *k-th Nearest Neighbor*: Uses the kth nearest neighbor distances to normalize pairwise distances and improve stability over SING.
3. **avgknn**  
   - *Average of k Nearest Neighbors*: Normalizes distances by the mean distance to the k nearest neighbors.
4. **avgknn_angle**  
   - *Average of k Nearest Neighbors with Angle*: Extends avgknn with an anisotropic term, similar to KA-SING’s approach.
5. **SING**  
   - *Original Implementation*: Taken from [di-marin/SING](https://github.com/di-marin/SING), uses nearest neighbor–based stability analysis with TDA-based parameter tuning.
6. **DBSCAN**  
   - *Density-Based Spatial Clustering*: Classic algorithm that identifies core points based on \(\epsilon\) and a minimum number of neighbors.
7. **OPTICS**  
   - *Ordering Points to Identify the Clustering Structure*: A variant of DBSCAN that captures density-based structure across multiple scales.

---

## Example Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/Stability-Incorporating-Neighborhood-Graph-with-Anisotropy.git
   cd Stability-Incorporating-Neighborhood-Graph-with-Anisotropy
