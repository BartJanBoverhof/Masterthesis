"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description Loading the data and training all networks.
"""



################### 0. Prerequisites ###################
#Loading packages
import torch #PyTorch deep-learning library
from torch import nn, optim #PyTorch additionals and training optimizer
import torch.nn.functional as F #PyTorch library providing a lot of pre-specified functions
import os
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


"""
try: #Importing network
    from pytorch.networks import Network
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)


################### 1. Loading data ###################
#Loading pickles
op = pickle.load(open("pipeline/prepared_data/p10op.pickle", "rb"))
en = pickle.load(open("pipeline/prepared_data/p10en.pickle", "rb"))
ta = pickle.load(open("pipeline/prepared_data/p10ta.pickle", "rb"))
"""


################### 1. PPG ###################
#Create datasetclass
class PytorchDataset(Dataset):
    """ PPG dataset """
    
    def __init__(self, datapath, modality):
        self.fulldata = pickle.load(open(datapath, "rb"))
        self.data = self.fulldata[modality]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


datasetje = PytorchDataset(datapath = "pipeline/prepared_data/p10op.pickle", modality = "PPG")



#Loading
trainloader = torch.utils.data.DataLoader(datasetje, 
                                          batch_size = 64, 
                                          shuffle = True)

next(iter(trainloader))[1]











###### TEMPORARY ########
from collections import Counter
#Inspecting data
#EEG
eeg_l = []
for i in ta["EEG"]:
    eeg_l.append(len(i))

Counter(eeg_l)
len(eeg_l)

plt.hist(x=eeg_l)
plt.show()






#PPG
ppg_l = []
for i in ta["PPG"]:
    ppg_l.append(len(i))

Counter(ppg_l)
len(ppg_l)

plt.hist(x=ppg_l)
plt.show()






#GSR
gsr_l = []
for i in ta["GSR"]:
    gsr_l.append(len(i))

Counter(gsr_l)
len(gsr_l)

plt.hist(x=gsr_l)
plt.show()
