# The Realtime Assesment of Mental Workload by Means of Multiple Bio-Signals.


**Masterthesis repository by Bart-Jan Boverhof.**  
This repository contains all the relevant code/files for the research project *Predicting Mental Workload: A Multimodal Intermediate Fusion Deep Learning Approach*. For any questions please contact bjboverhof@gmail.com. For the internal structure of the repository to work properly, it is important that the main folder is (this folder) is set as the working directory. There is an internal co-dependence in between the `1.pipeline`, `2.hpo`, `3.pytorch` subparts of this repository, in the sense that replication should start at 1 and proceed towards 3. The output of subpart `3.pytorch` are the network predictions, which are provided in this repository, such that the replicator does not need to train all networks and conduct hpo in order to replicate the final results. 

A general overview of the content and structure of this repository, in addition to the required software is provided in the following table:

| Folder | Description | Utilized software |
| ----------- | ----------- | ----------------- |
| `0.manuscript` | Written LaTeX manuscript | ![Texmaker](https://img.shields.io/badge/Texmaker-%35.0-orange) |
| `1.pipeline` | Folder that contains scripts to read data and cut into windows of `pytorch tensor` type. | ![Python version](https://img.shields.io/badge/Python-%33.9-yellow) [PyTorch version](https://img.shields.io/badge/PyTorch-%31.7-green)|
| `2.hpo` | Scripts to conduct hyperparamater optimization | ![Python version](https://img.shields.io/badge/Python-%33.9-yellow) ![PyTorch version](https://img.shields.io/badge/PyTorch-%31.7-green) ![Optuna](https://img.shields.io/badge/Optuna-%32.7.0-blue)| 
| `3.pytorch` | Definition of deep-learning models and scripts to train them and test them to obtain predictions | ![Python version](https://img.shields.io/badge/Python-%33.9-yellow) ![PyTorch version](https://img.shields.io/badge/PyTorch-%31.7-green) | 
| `4.results` | Scripts to create resulting statistics and plots from the obtained predictions | ![Python version](https://img.shields.io/badge/Python-%33.9-yellow) ![PyTorch version](https://img.shields.io/badge/PyTorch-%31.7-green) | 

More detailed information regarding the content of each of the subsection is provided in additional `README.md` files situated in the respective subsections.
---
