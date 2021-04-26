# Pipeline

### Subfolder description
The data obtained from *LSL* (see manuscript for an elaboration on *LSL*) is in `.xdf` format. Since files with `.xdf` extensions can't be directly read into Python, this subsection contains scripts to read the data, and write it into `.pandas dataframe` format. Subsequently, the time-series are cut into windows. 

### Subfolder content
This subsection of the repository contains the following objects: 
* `transformer.py`: Script that provides functions utilized for cutting epochs and writing data in usable format:
    - `load_data`: read the data from `.xdf` extension.
    - `create_windows`: determine timestamps for the selected time window.
    - `create_dataframe`: creates `pandas dataframes` based on the previously selected windows.
    - `cut_epochs`: cuts the data into epochs based on the selected time windows.
    - `transformer`: function that combines all of the above functions. 
* `main.py`: central script from which the data transformation is conducted.
* `raw_data (folder)`: folder that contains the raw daat to be transformed (not posted on GitHub)
* `prepared_data (folder)`: folder that contains the prepared data resulting from the above mentioned scripts (not posted on GitHub).

### Subfolder objective
Ultimately, the objective of this script is to write and save a total of 27 `.pickle` files, i.e. one file per participant. Each of these files contains a `dict` object with 6 keys reflecting:
* `EEG`: EEG cut windows.
* `PPG`: PPG cut windows.
* `EEG`: GSR cut windows.
* `EEG labels`: labels of the EEG windows.
* `PPG labels`: labels of the PPG windows.
* `GSR labels`: labels of the GSR windows.
