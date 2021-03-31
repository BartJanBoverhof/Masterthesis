######################################################
################## 0. PREREQUISITES ##################
######################################################
#Setting seed
set.seed(7931)

#Loading libraries
require(rstudioapi) #Utilized for automatically setting wd
require(ggplot2)  #Utilized for plotting

#Setting working directory
setwd(dirname(getActiveDocumentContext()$path)) 

#Loading data
data <- readRDS("./data/data.rds")

######################################################
################# 2. DATA PREPARATION ################
######################################################

#Create ordinal age variable
data$age_cat <- cut(data$age, breaks = c(14,19,24,29,34,39,44,49,54,59,64), labels = c("15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64"))

#Drop unused factor levels
data$gender <-  droplevels(data$gender)

######################################################
################### 3. CREATING PLOT #################
######################################################

#Creating plots
#Histogram
ggplot(data, aes(x = age_cat, fill=gender, color = gender))+
  geom_histogram(alpha=0.5, stat = "count",position="identity")+
  xlab("Age")+
  ylab("Count")

                 