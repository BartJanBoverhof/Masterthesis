# 5. Pipeline
The data obtained from *LSL* (see manuscript for an elaboration on *LSL*) is in `.xdf` format. Since files with `.xdf` extensions can't be directly read into Python, this subsection contains scripts to read and write the data as dataframe. The time-series are cut into windows. Ultimately, the goal of this script is to obtain a seperate dataframe for each window, for each modality seperately. Due to privacy concerns the data is not included provided.

For an extensive elaboration on the approach, please consult the research manuscript [manuscript.pdf](https://github.com/BartJanBoverhof/Masterthesis/tree/main/1.latex_manuscript).

---

![Status](https://img.shields.io/static/v1?label=Code+Status&message=Unfinished+and+Unexcecutable&color=red) 

---

This subsection of the repository contains the following objects: 
* `transformer`: Script that provides a function to read and write the data in a usable extension (`pandas dataframe`).

In addition to the earlier listed software, specifically the following packages are utilized:  
- `pyxdf`: package for reading .xdf files.
- `pandas`: data science package. 