# 2. Descriptives
This subsection of the repository deals with the description of the demographics information of the respondents. These demographics are reported in the manuscript, which is the file `manuscript.pdf` found in the folder `~/Masterthesis/1.latex_manuscript/`. 

---

![Status](https://img.shields.io/static/v1?label=Code+Status&message=Complete+and+Excecutable&color=<COLOR>) 

---

This subsection of the repository contains the following objects: 
* `1.create_table.R`: This script reads the data and creates a demographics table.
* `2.create_figure.R`: This script reads the data and creates a demographics histogram.
* `data` folder: contains the demograhpics data file (`data.rds`).

The following packages are used:  
* `rstudioapi `[package](https://cran.rstudio.com/web/packages/rstudioapi/index.html) for setting relative path.
* `xtable` [package](http://xtable.r-forge.r-project.org/) for creating LaTeX table. 
* `ggplot2` [package](https://cran.r-project.org/web/packages/ggplot2/index.html) for creating histogram. 
