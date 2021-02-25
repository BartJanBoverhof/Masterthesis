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
    import dataprep, networks
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)

participants = ["bci10", "bci12", "bci13", "bci17", "bci20", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", "bci38", 
                "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]


def TrainLoop(participant, modality, filters, hidden_dim, n_layers, drop, epochs, trainn):
    ###########################################################################################
    ########################## 1. Create PyTorch dataset & Loader(s) ##########################
    ###########################################################################################
    #Create PyTorch dataset definition class
    path = "pipeline/prepared_data/"+participant+"/data.pickle"
    pydata =  dataprep.PytorchDataset(path = path,       #Creating PyTorch dataset
                                      modality = modality)

    #Pre-specification of relevant paramaters
    batch_size = 25
    test_split = .2
    validation_split = .2


    padding_length = dataprep.PytorchDataset.__PaddingLength__(pydata) #Determining the longest window for later use
    dataprep.BatchTransformation.transfer([padding_length, modality]) #Transfer max padding length & modality vars to BatchTransfor class

    np.random.seed(3791)
    torch.manual_seed(3791)

    #Test split
    indices = list(range(len(pydata.dat))) #Create list of indices
    train_test_split = int(np.floor(test_split * len(pydata))) #Calculate number of windows to use for train/test
    np.random.shuffle(indices) #Shuffle indices.
    train_indices, test_indices = indices[train_test_split:], indices[:train_test_split] #Splitted indices.

    train_valid_split = int(np.floor(validation_split * len(train_indices))) #Calculate number of windows to use for val/train
    val_train_indices, val_indices = train_indices[train_valid_split:], train_indices[:train_valid_split] #Splitted indices.

    #Defining samplers
    test_sampler = SubsetRandomSampler(test_indices) #Train sampler
    train_sampler = SubsetRandomSampler(val_train_indices) #Train sampler
    valid_sampler = SubsetRandomSampler(val_indices) #Validation sampler
    
    #Defining loaders
    testloader = torch.utils.data.DataLoader(pydata, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= False,
                                        sampler = test_sampler,
                                        collate_fn = dataprep.BatchTransformation())
    validloader = torch.utils.data.DataLoader(pydata, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            sampler = valid_sampler,                                          
                                            collate_fn = dataprep.BatchTransformation())

    trainloader = torch.utils.data.DataLoader(pydata, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            sampler = train_sampler,
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
    
    print(model)

    #Loss function & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)




    ###########################################################################################
    ########################## 3. Training & Validation loop ##################################
    ###########################################################################################
    
    if trainn == True:
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


                ###################
                ##Validation loop##
                ###################
                model.eval()
                for windows, labels in validloader:
                    
                    #Validation pass
                    out = model(windows)
                    loss = criterion(out.squeeze(), labels)
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




        plt.plot(list(range(epochs-7)), train_list[7:len(train_list)], label = "train")
        plt.plot(list(range(epochs-7)), valid_list[7:len(valid_list)], label = "validation")
        plt.show()
        



    ###########################################################################################
    ########################## 4. Assessing model performance ##################################
    ###########################################################################################
    elif trainn == False:
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
                
            train_loss = test_loss/len(testloader.sampler)
            sum(diff)/len(testloader.sampler)
            print("end")


