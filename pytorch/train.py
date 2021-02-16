#!/usr/bin/env python
"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description Loading the data and training all networks.
"""



################### 0. Prerequisites ###################
#Loading packages
import torch 
from torch import optim #PyTorch additionals and training optimizer
import torch.nn as nn
import torch.nn.functional as F #PyTorch library providing a lot of pre-specified functions
import os
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import optim

try: #Importing network
    from networks import PpgNet
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)


################### Create PyTorch dataset ###################
#Create datasetclass
class PytorchDataset(Dataset):
    
    def __init__(self, path, modality):
        
        self.dat = pickle.load(open(path, "rb")) #Open pickle
        self.labels = self.dat["labels"] #Saving labels
        self.dat = self.dat[modality] #Saving only modality of intrest 

        #Cutting stored epochs into same size
        self.lengths = [] 
        for i in self.dat: #Determining lowest length tensor
            x = i.shape[1]
            self.lengths.append(x)

        lowest = min(self.lengths)

        self.counter = 0
        for i in self.dat: #Reshaping tensors
            if len(i) != lowest:
                self.dat[self.counter] = i.narrow(1,0,lowest)
            self.counter +=1
        


    def __len__(self):
        return len(self.dat)


    def __getitem__(self, idx):
        windows = self.dat[idx]
        labels = self.labels[idx]

        return windows, labels

#Creating dataset and trainload
pydata = PytorchDataset(path = "pipeline/prepared_data/bci10/data.pickle", 
                        modality = "PPG")

trainloader = torch.utils.data.DataLoader(pydata, 
                                          batch_size = 25, 
                                          shuffle = True)


#define model
model = PpgNet()
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


#### TRAIN #####
epochs = 100

for epoch in range(1, epochs+1):
    running_loss = 0
    
    model.train()
    for windows, labels in trainloader:

        #Training pass
        optimizer.zero_grad()

        out = model(windows)
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