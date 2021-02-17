# Pipeline
The data obtained from *LSL* (see manuscript for an elaboration on *LSL*) is in `.xdf` format. Since files with `.xdf` extensions can't be directly read into Python, this subsection contains scripts to read and write the data as dataframe. The time-series are cut into windows. Ultimately, the goal of this script is to obtain a seperate dataframe for each window, for each modality seperately. Due to privacy concerns the data is not included provided.

For an extensive elaboration on the approach, please consult the research manuscript [manuscript.pdf](https://github.com/BartJanBoverhof/Masterthesis/tree/main/manuscript).

---

![Status](https://img.shields.io/static/v1?label=Code+Status&message=Unfinished+and+Unexcecutable&color=red) 

---

This subsection of the repository contains the following objects: 
* `transformer.py`: Script that provides functions utilized for cutting epochs and writing data in usable format:
    - `load_data`: read the data from `.xdf` extension.
    - `create_windows`: determine timestamps for the selected time window.
    - `create_dataframe`: creates `pandas dataframes` based on the previously selected windows.
    - `cut_epochs`: cuts the data into epochs based on the selected time windows.
    - `transformer`: function that combines all of the above functions. 
* `main.py`: central script from which the data transformation is conducted.

In addition to the earlier listed software, specifically the following packages are utilized:  
- `pyxdf`: package for reading .xdf files.
- `pandas`: data science package. 
- `pickle`: transform data is .pickle file.


