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
    """ PPG dataset """
    
    def __init__(self, op_path, en_path, ta_path, modality):
        self.opdat = pickle.load(open(op_path, "rb"))
        self.endat = pickle.load(open(en_path, "rb"))
        self.tadat = pickle.load(open(ta_path, "rb"))

        #Extract labels
        self.oplabels = self.opdat["labels"]
        self.enlabels = self.endat["labels"]
        self.talabels = self.tadat["labels"]
        
        #Save only modality of intrest 
        self.opdat = self.opdat[modality]
        self.endat = self.endat[modality]
        self.tadat = self.tadat[modality]

        #Concatenate
        self.alldat = self.opdat + self.endat + self.tadat
        self.alllabels = torch.cat((self.oplabels, self.enlabels, self.talabels), 0)
        
        #Cutting stored epochs into same size
        #Determining lowest length tensor
        self.lengths = []
        for i in self.alldat:
            x = i.shape[1]
            self.lengths.append(x)

        lowest = min(self.lengths)

        #Reshaping tensors
        self.counter = 0
        for i in self.alldat:
            if len(i) != lowest:
                self.alldat[self.counter] = i.narrow(1,0,lowest)
            self.counter +=1
        


    def __len__(self):
        return len(self.opdat) + len(self.endat) + len(self.tadat)


    def __getitem__(self, idx):
        windows = self.alldat[idx]
        labels = self.alllabels[idx]

        return windows, labels

#Creating dataset and trainload
pydata = PytorchDataset(op_path = "pipeline/prepared_data/p10op.pickle", 
                        en_path = "pipeline/prepared_data/p10en.pickle", 
                        ta_path = "pipeline/prepared_data/p10ta.pickle", 
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