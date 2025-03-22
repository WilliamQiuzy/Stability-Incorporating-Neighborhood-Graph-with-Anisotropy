# avgknn: Average k Nearest Neighbors Clustering

This directory provides an implementation of the **avgknn** approach, a variant of the SING framework that uses the average distance to the \(k\) nearest neighbors for local scale estimation. This method is particularly useful for datasets where a single kth nearest neighbor distance may not sufficiently capture local density variations.

## Usage

```bash
python main.py --filename <input_file> --filetype <data_type> --k <num_neighbors> --epsilon <epsilon_value> --density <density_exponent> --drawEdges
