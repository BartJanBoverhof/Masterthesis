#!/usr/bin/env python
"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description Loading the data and training all networks.
"""



################### 0. Prerequisites ###################
#Loading packages
import torch 
from torch import nn, optim #PyTorch additionals and training optimizer
import torch.nn.functional as F #PyTorch library providing a lot of pre-specified functions
import os
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import optim

"""
try: #Importing network
    from pytorch.networks import Network
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)
"""


################### Create PyTorch dataset ###################
#Create datasetclass
class PytorchDataset(Dataset):
    """ PPG dataset """
    
    def __init__(self, datapath, modality):
        self.fulldata = pickle.load(open(datapath, "rb"))
        self.data = self.fulldata[modality]
        self.labels = self.fulldata["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        epochs = self.data[idx]
        labels = self.labels[idx]

        return epochs, labels

#Creating dataset and trainload
pydata = PytorchDataset(datapath = "pipeline/prepared_data/p10op.pickle", modality = "PPG")

trainloader = torch.utils.data.DataLoader(pydata, 
                                          batch_size = 64, 
                                          shuffle = True)




#labels_int = labels.type(torch.int)

#TRAIN Model
model = nn.Sequential(nn.Linear(1279, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 20),
                      nn.Linear(20,1), 
                      nn.Sigmoid())


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

iterations = 30
for i in range(iterations):
    running_loss = 0
    
    for epochs, labels in trainloader:
        #Flatten
        epoch_flat  = epochs.view(epochs.shape[0], -1)

        #Training pass
        optimizer.zero_grad()

        out = model(epoch_flat)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

'''
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
'''