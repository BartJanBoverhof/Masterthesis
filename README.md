# The Realtime Assesment of Mental Workload by Means of Multiple Bio-Signals.
Masterthesis repository by Bart-Jan Boverhof. 
This repository contains all relevant code/files for the research project *The Realtime Assesment of Mental Workload by Means of Multiple Bio-Signals* - by Bart-Jan Boverhof. 
For any questions please contact bjboverhof@gmail.com.
A general overview of the content and structure of this repository, in addition to the required software is provided in the following table:

| File/Folder | Description | Required software |
| ----------- | ----------- | ----------------- |
| `1.latex_manuscript` | Written manuscript LaTeX-repository | Any LaTeX engine ([Texmaker](https://www.xm1math.net/texmaker/) recommended) |
| `2.descriptives` | R code & data to obtain descriptive statistics as reported in the manuscript (method section) | [R](https://www.r-project.org/) (version 3.6.2) |
| `3.main_analysis` | R code & data to conduct the main analysis reported in the manuscript (result section) | [Python](https://www.python.org/) (version 3.9.1) & [PyTorch](https://pytorch.org/) | 

More detailed information on the content within each of the subsection is provided via additional `README.md` files. Each of the three subsection is accompanied with one additional `README.md` file.

---
### Some notes to Gerko: 
- As of yet I don't have access to the main data files yet. Therefore, the subfolder `3.main_analysis` is work in progress. In order for you to get an idea about the future repository structure, I did already set up some initial files/projects/structure in this folder. 
- On the main page, you might observe that Jan-Willem is a contributer. The reason for this is that I wanted to check whether someone invited to this repository could commit directly to the main branch. Well, apparently they can... I removed the commits from the git history, however Jan-Willem is still listed as contributer on the main page of the project. I did some research to find out how to reverse this, but couldn't find anything that worked. If you happen to have an idea, im glad to hear it. Just removing the commits from the history didn't do the trick. Also, it seems to be impossible to grant read-only rights on GitHub (I by the way also contacted the GitHub customer service to see if they can help).
- The complete git history of the LaTeX manuscript (in the `1.latex_manuscript` subfolder) does not date back fully to the start of this manuscript, but rather initiates slightly later in time. The reason for this is that I initially used overleaf, and as a consequence I didn't log git history of the document. Therefore, only the very early work on the manuscript was not logged via git. 
