#!/usr/bin/env python
"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description Loading the data and training all networks.
"""


###########################################################################################
###################################### 0. Prerequisites ###################################
###########################################################################################
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
    import networks, dataprep
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)

participants = ["bci10", "bci12", "bci13", "bci17", "bci20", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", "bci38", 
                "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]

###########################################################################################
########################## 1. Create PyTorch dataset & Loader(s) ##########################
###########################################################################################
#Create PyTorch dataset definition class
path = "pipeline/prepared_data/"+participants[0]+"/data.pickle"
pydata =  dataprep.PytorchDataset(path = path,       #Creating PyTorch dataset
                                  modality = "PPG")

#Pre-specification of relevant paramaters
batch_size = 25
test_split = .2

validation_split = .2
validation = True


padding_length = dataprep.PytorchDataset.__PaddingLength__(pydata) #Determining the longest window for later use
dataprep.BatchTransformation.transfer(padding_length) #Transfer max padding length var to BatchTransfor class

np.random.seed(3791)
torch.manual_seed(3791)

#Test split
indices = list(range(len(pydata.dat))) #Create list of indices
train_test_split = int(np.floor(test_split * len(pydata))) #Calculate number of windows to use for train/test
np.random.shuffle(indices) #Shuffle indices.
train_indices, test_indices = indices[train_test_split:], indices[:train_test_split] #Splitted indices.

testloader = torch.utils.data.DataLoader(pydata, #Test loader for later use
                                    batch_size = len(test_indices), 
                                    shuffle = False,
                                    sampler = test_indices,
                                    collate_fn = dataprep.BatchTransformation())



#Validation split
if validation == True: 
    train_valid_split = int(np.floor(validation_split * len(train_indices))) #Calculate number of windows to use for val/train
    val_train_indices, val_indices = train_indices[train_valid_split:], train_indices[:train_valid_split] #Splitted indices.

    #Defining samplers
    train_sampler = SubsetRandomSampler(val_train_indices) #Train sampler
    valid_sampler = SubsetRandomSampler(val_indices) #Validation sampler

    validloader = torch.utils.data.DataLoader(pydata, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            sampler = valid_sampler,                                          
                                            collate_fn = dataprep.BatchTransformation())

    #Defining loaders
    trainloader = torch.utils.data.DataLoader(pydata, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            sampler = train_sampler,
                                            collate_fn = dataprep.BatchTransformation())
else:
    
    train_sampler = SubsetRandomSampler(train_indices) #Train sampler

    trainloader = torch.utils.data.DataLoader(pydata, #Training loader
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        sampler = train_indices,
                                        collate_fn = dataprep.BatchTransformation())





###########################################################################################
################################## 2. Defining model ######################################
###########################################################################################
#Obtaining network
model = networks.PpgNet()
print(model)

#Loss function & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)




###########################################################################################
########################## 3. Training & Validation loop ##################################
###########################################################################################
#Prerequisites
epochs = 50
train_list = []
valid_list = []
valid_loss_min = np.Inf

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
    if validation == True:

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
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min, valid_loss))

            torch.save(model, "pytorch/trained_models/"+participants[0]+"_PPG.pt")
            valid_loss_min = valid_loss


    else: 
        
        #Averages losses
        train_loss = train_loss/len(trainloader.sampler)
        train_list.append(train_loss)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss))

plt.plot(list(range(epochs-1)), train_list[1:len(train_list)], label = "train")
plt.plot(list(range(epochs-1)), valid_list[1:len(valid_list)], label = "validation")
plt.show()




###########################################################################################
########################## 4. Assessing model performance ##################################
###########################################################################################
trained_model = torch.load("pytorch/trained_models/"+participants[0]+"_PPG.pt")

test_loss = 0.0

model.eval()
for windows, labels in testloader:
    
    #Test pass
    out = model(windows)
    loss = criterion(out, labels)
    test_loss += loss.item()*windows.size(0)
    avg_loss = test_loss / out.shape[0]

    difference = labels - torch.squeeze(out)