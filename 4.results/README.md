# Results

### Subfolder description
This subsection of the repository contains all code with which results provided troughout the manuscript are created, including test statistics and plots. 

### Subfolder content
This subsection of the repository contains the following objects: 
* `results.py`: Contains all functions to re-create results provided in the manusript. 
* `predictions (folder)`: Contains the predictions of train models as resulting from repository subparts 1-3. These predictions are provided, in case the replicator wishes to assess results without going through the lengthy and computationally intensive process of preparing the data, training the models and conducting hpo. 
* `descriptives (folder`: Contains the data for the descriptives as reported in the manuscript.

### Subfolder objective
The objective of this subfolder is firstly to train all networks. This should be conducted from the `train.py` file. After all networks are trained, and stored in the `trained_models folder`, the testing on the test-partition can be conducted with the `predict.py` script. This script outputs a single `.pickle` object per network architecture, containing the aggregation over all respondents of the predicted test windows, which are stored in the repository subfolder `4.results`.