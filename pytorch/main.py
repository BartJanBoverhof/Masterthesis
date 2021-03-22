import numpy as np
import torch
import os

try: #Importing network
    import train_single, train_multi, dataprep
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'.")
    print("Current working directory is:", wd)


participants = ["bci10", "bci12", "bci13", "bci17", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", 
                "bci38", "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]

modalities = ["EEG","PPG","GSR"]


#############################################################################################
#############################################################################################
################################### Single-modality networks ################################
#############################################################################################
#############################################################################################

drop = 0.25
epochs = 50
batch_size = 10
trainortest = "train"
np.random.seed(3791)
torch.manual_seed(3791)
"""
for i in participants:
    for modality in modalities:

        train_single.TrainLoop(participant = i, modality = modality,
                        drop = drop, batch_size = batch_size, 
                        epochs = epochs, trainortest = trainortest)
"""


#############################################################################################
#############################################################################################
################################### Multi-modality networks #################################
#############################################################################################
#############################################################################################

train_multi.TrainLoop(participant = "bci13",
                drop = drop, batch_size = batch_size, 
                epochs = epochs, trainortest = trainortest)









"""
#TEMP
#check amount of data for each person
lengths = np.array([])
for participant in participants:
    path = "pipeline/prepared_data/"+participant+".pickle"
    pydata =  dataprep.PytorchDataset(path = path,           #Creating PyTorch dataset
                                      modality = "EEG")
    padding_length = dataprep.PytorchDataset.__PaddingLength__(pydata) #Determining the longest window for later use
    lengths = np.append(lengths, [padding_length])
import matplotlib.pyplot as plt
plt.plot(lengths)
plt.show()
print("ha;;p")
"""
