# avgknn_angle: Average k Nearest Neighbors with Angle

This subdirectory provides an implementation of the average k nearest neighbors (avgknn) clustering method extended with an angular adjustment term. This algorithm refines the normalized proximity measure by accounting for the local orientation of data clusters, thereby better capturing anisotropic structures and irregular sampling in the dataset.

## Usage

```bash
python main.py --filename <input_file> --filetype <data_type> --k <num_neighbors> --epsilon <epsilon_value> --q <exponent> --gamma <gamma_value> [--drawEdges]
