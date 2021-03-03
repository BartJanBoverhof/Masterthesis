import numpy as np
import torch
import os

try: #Importing network
    import train, dataprep
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)


participants = ["bci10", "bci12", "bci13", "bci17", "bci20", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", 
                "bci38", "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]

drop = 0.25
epochs = 50
trainortest = "test"
np.random.seed(3791)
torch.manual_seed(3791)


train.TrainLoop(participant = participants[18], modality = "EEG",
                drop = drop,
                epochs = epochs, trainortest = trainortest)
