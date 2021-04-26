# Hpo

### Subfolder description
The scripts in this subfolder conduct hyperparamater optimization by means of `optuna` toolbox. It does so for each respondent, and for each of the four models seperately. For each of the 108 (27 paricipants x 4 networks) seperately optimized networks, a set of optimized parameters is stored as `.pickle` format file. Per network architecture, these sets of paramaters are subsequently combined into one file, resulting in a total of 4 files (one for each architecture), each containing the sets of optimized paramaters for each respondent of that particulair network architecture. 

### Subfolder content
This subsection of the repository contains the following objects: 
* `hpo_eeg.py`: script that conducts hpo for the unimodal EEG network.
* `hpo_ppg.py`: script that conducts hpo for the unimodal PPG network.
* `hpo_gsr.py`: script that conducts hpo for the unimodal GSR network.
* `hpo_multi.py`: script that conducts hpo for the multimodal network.
* `combine.py`: script that combines the optimized hyperparamaters into a single file per network architecture.
* `hyper_parameters_notcombined (folder)`: folder in which optimized sets of hyperparamaters are stored (not posted on GitHub).
* `hyper_parameters (folder)`: folder in which combined optimized sets of hyperparamaters are stored (not posted on GitHub).

