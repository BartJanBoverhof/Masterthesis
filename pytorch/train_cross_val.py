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
from sklearn.model_selection import KFold


try: #Importing network
    import dataprep, networks
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)


def TrainLoop(participant, modality, filters, hidden_dim, n_layers, drop, epochs, trainortest):
    
    ###########################################################################################
    ########################## 1. Create PyTorch dataset & Loader(s) ##########################
    ###########################################################################################
    #Create PyTorch dataset definition class
    path = "pipeline/prepared_data/"+participant+"/data.pickle"
    pydata =  dataprep.PytorchDataset(path = path,       #Creating PyTorch dataset
                                      modality = modality)

    padding_length = dataprep.PytorchDataset.__PaddingLength__(pydata) #Determining the longest window for later use
    dataprep.BatchTransformation.transfer([padding_length, modality]) #Transfer max padding length & modality vars to BatchTransfor class
               
               
    #Making splits
    batch_size = 10
    test_split = .1

    ################
    ## TEST SPLIT ##
    ################ 
    
    indices = list(range(len(pydata.dat))) #Create list of indices
    train_test_split = int(np.floor(test_split * len(pydata))) #Calculate number of windows to use for train/test
    np.random.shuffle(indices) #Shuffle indices.
    train_indices, test_indices = indices[train_test_split:], indices[:train_test_split] #Splitted indices.

    test_sampler = SubsetRandomSampler(test_indices) #Train sampler    
    testloader = torch.utils.data.DataLoader(pydata, #Test loader 
                                    batch_size = batch_size, 
                                    shuffle = False,
                                    drop_last= False,
                                    sampler = test_sampler,
                                    collate_fn = dataprep.BatchTransformation()) 
    ######################
    ## VALIDATION SPLIT ##
    ######################
    k = 5
    train = {}

    kfold = KFold(n_splits=k, shuffle=True)

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_indices)):
        print("x")
        print('--------------------------------')
        print(f'FOLD {fold+1}')
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
        
        trainloader = torch.utils.data.DataLoader(pydata, #Training loader
                                                batch_size = batch_size, 
                                                shuffle = False,
                                                drop_last= False,
                                                sampler = train_subsampler,
                                                collate_fn = dataprep.BatchTransformation())

        validloader = torch.utils.data.DataLoader(pydata, #Test loader 
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            sampler = valid_subsampler,
                                            collate_fn = dataprep.BatchTransformation())
                                            

        ###########################################################################################
        ################################## 2. Defining model ######################################
        ###########################################################################################
        #Defining network
        if modality == "PPG":
            model = networks.PPGNet(filters, hidden_dim, n_layers, drop)
        elif modality == "GSR":
            model = networks.GSRNet(filters, hidden_dim, n_layers, drop)
        elif modality == "EEG":
            model = networks.EEGNet(filters, drop)
        

        #Loss function & Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)


        ###########################################################################################
        ########################## 3. Training & Validation loop ##################################
        ###########################################################################################
        
        if trainortest == "train":
            #Prerequisites
            clip = 5
            train_list = []
            valid_list = []
            valid_loss_min = np.Inf

            ######################
            ### PPG & GSR LOOP ###
            ######################
            if modality == "PPG" or modality == "GSR": #If the network includes an LSTM layer (holds for PPG & GSR) 
                for epoch in range(1, epochs+1):
                        
                    train_loss = 0.0
                    valid_loss = 0.0

                    #####################
                    ### Training loop ###
                    #####################
                    h = model.InitHiddenstate(batch_size)
                    model.train()
                    for windows, labels in trainloader:

                        h = tuple([e.data for e in h])
                        #Training pass
                        optimizer.zero_grad()
                        out, h = model(windows, h)
                        loss = criterion(out.squeeze(), labels)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), clip)
                        optimizer.step()
                        train_loss += loss.item() * windows.size(0)


                    #####################
                    ## Validation loop ##
                    #####################
                    h = model.InitHiddenstate(batch_size)
                    model.eval()
                    for windows, labels in validloader:
                        
                        h = tuple([each.data for each in h])

                        #Validation pass
                        out, h = model(windows, h)
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

                            torch.save(model, "pytorch/trained_models/"+participant+"_"+modality+".pt")
                            valid_loss_min = valid_loss

            ################
            ### EEG LOOP ###
            ################
            elif modality == "EEG": #If the network includes an LSTM layer (holds for PPG & GSR) 
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
                        loss = criterion(out.squeeze(), labels)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * windows.size(0)
                    
                    train_loss = train_loss/len(trainloader.sampler)
                    print('Epoch: {} \tTrainLoss: {:.6f}'.format(
                    epoch, train_loss))

                ###################
                ##Validation loop##
                ###################
                model.eval()
                for windows, labels in validloader:
                    
                    #Validation pass
                    out = model(windows)
                    loss = criterion(out.squeeze(), labels)
                    valid_loss += loss.item()*windows.size(0)


                valid_loss = valid_loss/len(validloader.sampler)
                

                print('VallLoss: {:.6f}'.format(valid_loss))
                    
        




        ###########################################################################################
        ########################## 4. Assessing model performance ##################################
        ###########################################################################################
        elif trainortest == "test":
            trained_model = torch.load("pytorch/trained_models/"+participant+"_"+modality+".pt")

            test_loss = 0.0

            if modality == "GSR" or modality == "PPG":
                h = model.InitHiddenstate(batch_size)

                model.eval()
                for windows, labels in testloader:
                    
                    #Test pass    
                    h = tuple([each.data for each in h])
                    out, h = model(windows, h)
                    loss = criterion(out.squeeze(), labels)
                    test_loss += loss.item()*windows.size(0)

                    difference = labels - torch.squeeze(out)

            elif modality == "EEG":
                
                model.eval()
                diff = [] 

                for windows, labels in testloader:
                    
                    #Test pass    
                    out = model(windows)
                    loss = criterion(out.squeeze(), labels)
                    test_loss += loss.item()*windows.size(0)

                    foo = (out.squeeze() - labels)
                    for x in foo:
                        diff.append(float(x))
                    
                test_loss = test_loss/len(testloader.sampler)
                sum(diff)/len(testloader.sampler)
                print("end")


