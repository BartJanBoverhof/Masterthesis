######################################################
################## 0. PREREQUISITES ##################
######################################################
#Setting seed
set.seed(7931)

#Loading libraries
require(rstudioapi) #Utilized for automatically setting wd
require(xtable)

#Setting working directory
setwd(dirname(getActiveDocumentContext()$path)) 

#Loading data
data <- readRDS("./data/data.rds")

######################################################
################# 2. DATA PREPARATION ################
######################################################

#Create ordinal age variable
data$age_cat <- cut(data$age, breaks = c(9,19,29,39,49,59,69), labels = c("10-19","20-29","30-39","40-49","50-59","60-69"))

#Drop unused factor levels
data$gender <-  droplevels(data$gender)

######################################################
################## 3. CREATING TABLE #################
######################################################

#Creating crosstab object
table <- table(data$age_cat, data$gender)

#Creating latex table object
print(xtable(table))