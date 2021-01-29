# The Realtime Assesment of Mental Workload by Means of Multiple Bio-Signals.


**Masterthesis repository by Bart-Jan Boverhof.**  
This repository contains all the relevant code/files for the research project *The Realtime Assesment of Mental Workload by Means of Multiple Bio-Signals*. For any questions please contact bjboverhof@gmail.com.  
A general overview of the content and structure of this repository, in addition to the required software is provided in the following table:

| Folder | Description | Utilized software |
| ----------- | ----------- | ----------------- |
| `descriptives` | R code & data to obtain descriptive statistics as reported in the manuscript (method section) | ![R version](https://img.shields.io/badge/R-%33.6-blue) |
| `manuscript` | Written LaTeX manuscript | ![Texmaker](https://img.shields.io/badge/Texmaker-%35.0-orange) |
| `pipeline` | Folder that contains scripts to read data and cut into epochs of `pandas df` type. | ![Python version](https://img.shields.io/badge/Python-%33.9-yellow) |
| `pytorch` | Definition of deep-learning models and scripts to train these | ![Python version](https://img.shields.io/badge/Python-%33.9-yellow) ![PyTorch version](https://img.shields.io/badge/PyTorch-%31.7-green) | 

More detailed information regarding the content of each of the subsection is provided in additional `README.md` files situated in the respective subsections.

---
### Some notes to Gerko: 
- The subfolder `pytorch` and `pipeline` are work in progress. In order for you to get an idea about the future repository structure, I did already set up some initial files/projects/structure in this folder. The structure may as of yet seem a bit messy / unlogical. The reason for this is that I can't clearly define the structure yet, for I don't know which kind of structure will work. 
- On the main page, you might observe that Jan-Willem is a collaborator. The reason for this is that I wanted to check whether someone invited to this repository could commit directly to the main branch. Well, apparently they can... I removed the commits from the git history, however Jan-Willem is still listed as collaborator on the main page of the project. I did some research to find out how to reverse this, but couldn't find anything that worked. If you happen to have an idea, im glad to hear it. Just removing the commits from the history didn't do the trick. (I by the way also contacted the GitHub customer service to see if they can help).
- The complete git history of the LaTeX manuscript (in the `manuscript` subfolder) does not date back fully to the start of this manuscript, but rather initiates slightly later in time. The reason for this is that I initially used overleaf, and as a consequence I didn't log git history of the document. Therefore, only the very early work on the manuscript was not logged via git. 
