# SING: Stability-Incorporated Neighborhood Graph

We provide the basic functionality of SING.

**Usage:**

python main.py --filename --filetype --epsilon --density --drawEdges

python main.py --filename='./examples/stipples/Tree.csv' --filetype='stipples' --density='1.1' --epsilon='0.44'


**Parameters:**
- filename: the input data, usually a list of 2D coordinates, sometimes with additional information per point e.g. radius
- filetype: th etype of input data, which can be 'stipples', 'disks' or 'species'. Stipples files only contain 2D coordinates of each point, disks files contain points which a given radius, while species files are exmaples from [Ecormier-Nocca et al. 2019], of points with radius and certain other properties which are ignored for our use case
- epsilon: (optional) the default value is 1.0
- density: (optional) the exponent for the density-aware variant of SING, default value is 0.0, which ignores the density component
- drawEdges: (optional) drawing the SING edges on top of the classes


The application plots the clustering using the given arguments, and the persistence diagram.

## License

This work is licensed under a  
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
