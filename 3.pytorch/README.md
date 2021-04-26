# PyTorch

### Subfolder description
This subsection of the repository contains all PyTorch related code. This includes network architectures utilized in the main analysis, as well as scripts that are utilized for training and predicting values with the trained networks.

### Subfolder content
This subsection of the repository contains the following objects: 
* `networks.py`: Script in which all four network architectures are defined.
* `loop_single.py`: Main training loop file for the three unimodal network architectures (`EEG`, `PPG` and `GSR`).
* `loop_multi.py`: Main training loop file for the multimodal network architecture.
* `train.py`: Main script from which all networks are trained.
* `predict.py`: Main script from which all trained network are utilized to predict on the test-partition data.
* `utility.py`: Utility functions that are utilized throughout the training/testing process. 

### Subfolder objective
The objective of this subfolder is firstly to train all networks. This should be conducted from the `train.py` file. After all networks are trained, and stored in the `trained_models folder`, the testing on the test-partition can be conducted with the `predict.py` script. This script outputs a single `.pickle`object per network architecture, containing the aggregation over all respondents of the predicted test windows, which are stored in the repository subfolder `4.results`.