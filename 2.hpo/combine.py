"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description This file combines all files resulting from the hpo search into four pickle file. 
"""

#Importing packages
import pickle

#Define participants
participants = ["bci10", "bci12", "bci13", "bci17", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", 
                "bci38", "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]

#Define modalities
modalities = ["EEG","PPG","GSR","multi"]

#Combine hyper-parameters
for modality in modalities:
    combination = {}
    for i in participants:
        x = pickle.load(open("hpo/hyper_parameters_notcombined/"+modality+"_"+i+".pickle", "rb"))
        combination[i] = x.params

    with open("hpo/hyper_parameters/"+modality+".pickle", 'wb') as handle: #Save as pickle
        pickle.dump(combination, handle, protocol=pickle.HIGHEST_PROTOCOL)


