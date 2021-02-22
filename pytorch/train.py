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
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np

try: #Importing network
    import networks
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)

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
        
        #Determining the longest window for later use
        lengths = []
        for i in self.dat:
            lengths.append(len(i))
        longest_window = max(lengths)

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
    
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value = 0) #Pad
    
        #Obtaining Sorted labels and standardizing them
        labels = []
        for i in sorted_batch:
            label = float(i[1])
            label = (label-1) / 20
            labels.append(label)
        labels = torch.tensor(labels)
        
        #TRANSPOSE BATCH 
        sequences_padded = torch.transpose(sequences_padded, 1, 2)

        return sequences_padded, labels




#Creating dataset and trainloader
pydata = PytorchDataset(path = "pipeline/prepared_data/bci17/data.pickle", 
                        modality = "PPG")



#Paramaters
#Determining the longest window for later use
lengths = []
for i in pydata.dat:
    lengths.append(len(i))
longest_window = max(lengths)

batch_size = 25
validation_split = .2
dataset_size = len(pydata.dat) #Obtain size dataset

#Split the data in train and test
indices = list(range(dataset_size)) #Create list of indices
split = int(np.floor(validation_split * dataset_size)) #Calculate number of windows to use for val/train
np.random.shuffle(indices) #Shuffle indices.
train_indices, val_indices = indices[split:], indices[:split] #Splitted indices.

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


trainloader = torch.utils.data.DataLoader(pydata, 
                                          batch_size = batch_size, 
                                          shuffle = False,
                                          sampler = train_sampler,
                                          collate_fn = BatchTransformation())

validloader = torch.utils.data.DataLoader(pydata, 
                                          batch_size = batch_size, 
                                          shuffle = False,
                                          sampler = valid_sampler,                                          
                                          collate_fn = BatchTransformation())








################### 2. Defining model ###################
model = networks.PpgNet()
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)






################### 3. Training  ###################
epochs = 50
train_list = []
valid_list = []

for epoch in range(1, epochs+1):
    
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    ###Training loop###
    ###################
    model.train()
    for windows, labels in trainloader:

        #Training pass
        optimizer.zero_grad()
        out = model(windows)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * windows.size(0)

    ###################
    ##Validation loop##
    ###################
    model.eval()
    for windows, labels in validloader:
        
        #Validation pass
        out = model(windows)
        loss = criterion(out, labels)
        valid_loss += loss.item()*windows.size(0)

    #Averages losses
    train_loss = train_loss/len(trainloader.sampler)
    valid_loss = valid_loss/len(validloader.sampler)

    train_list.append(train_loss)
    valid_list.append(valid_loss)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
plt.plot(list(range(epochs-1)), train_list[1:len(train_list)], label = "train")
plt.plot(list(range(epochs-1)), valid_list[1:len(valid_list)], label = "validation")
plt.show()
