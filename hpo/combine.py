import pickle

participants = ["bci10", "bci12", "bci13", "bci17", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", 
                "bci38", "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]

modality = "EEG"

combination = {}

for i in participants:
    x = pickle.load(open("hpo/hyper_parameters/"+modality+"_"+i+".pickle", "rb"))
    combination[i] = x

with open("hpo/hyper_parameters/"+modality+"_"+params+".pickle", 'wb') as handle: #Save as pickle
    pickle.dump(combination, handle, protocol=pickle.HIGHEST_PROTOCOL)