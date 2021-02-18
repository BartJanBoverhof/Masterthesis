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
from torch.nn.utils.rnn import pad_sequence

try: #Importing network
    from networks import PpgNet
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)

#Paramaters
batch_size = 5
train_prop = 0.8 

#Split the data in train and test


################### 1. Create PyTorch dataset & Loader ###################
#Create datasetclass
class PytorchDataset(Dataset):
    
    def __init__(self, path, modality):
        """
        Purpose: 
            Load pickle object and save only specified data. 
        """
        dat = pickle.load(open(path, "rb")) #Open pickle
        key = "labels_"+modality #Determining right dict key
        self.labels = dat[key] #Saving labels
        self.dat = dat[modality] #Saving only modality of interest 

    def __minmax__(self):
        #Obtain training set min & max boundaries
        maxlist = []
        minlist = []

        for i in self.dat:
            maxlist.append(torch.max(i))
            minlist.append(torch.min(i))
         
        minmax_boundaries = (float(min(minlist)), float(max(maxlist)))

        return (minmax_boundaries)

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        windows = self.dat[idx]
        labels = self.labels[idx]

        return windows, labels


class BatchTransformation:
    def __call__(self, batch):
        #PADDING
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True) #Sort batch in descending
        sequences = [x[0] for x in sorted_batch] #Get ordered windows

        means = []
        for tensor in sequences: #Obtain tensor means 
            means.append(float(torch.mean(tensor)))
    
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value =  (sum(means) / len(means))) #Pad
    
        #Obtaining sorted labels
        labels = []
        for i in sorted_batch:
            labels.append(float(i[1]))
        labels = torch.tensor(labels)
        
        #TRANSPOSE BATCH 
        sequences_padded = torch.transpose(sequences_padded, 1, 2)

        #MINMAX TRANSFORMATION
        normalized =  sequences_padded.clone()
        normalized = normalized.view(sequences_padded.size(0), -1)
        normalized -= MinMaxBoundaries[0]
        normalized /= (MinMaxBoundaries[1] - MinMaxBoundaries[0])
        normalized = normalized.view(sequences_padded.shape[0], sequences_padded.shape[1], sequences_padded.shape[2])

        return normalized, labels


#Creating dataset and trainloader
pydata = PytorchDataset(path = "pipeline/prepared_data/bci10/data.pickle", 
                        modality = "PPG")

MinMaxBoundaries = PytorchDataset.__minmax__(pydata) #Determine min-max boundaries training set

trainloader = torch.utils.data.DataLoader(pydata, 
                                          batch_size = 700, 
                                          shuffle = False,
                                          collate_fn = BatchTransformation())


################### 2. Defining model ###################
model = PpgNet()
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


################### 3. Training loop ###################
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