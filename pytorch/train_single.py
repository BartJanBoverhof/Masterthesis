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


def TrainLoop(participant, modality, drop, epochs, trainortest, batch_size):
    
    ###########################################################################################
    ########################## 1. Create PyTorch dataset & Loader(s) ##########################
    ###########################################################################################
    #Create PyTorch dataset definition class
    
    path = "pipeline/prepared_data/"+participant+".pickle"
    pydata =  dataprep.PytorchDataset(path = path,       #Creating PyTorch dataset
                                      modality = modality)

    padding_length = dataprep.PytorchDataset.__PaddingLength__(pydata) #Determining the longest window for later use
    dataprep.BatchTransformation.transfer([padding_length, modality]) #Transfer max padding length & modality vars to BatchTransfor class
               
               
    #Making splits
    test_split = .1
    validation_split = .1

    ################
    ## TEST SPLIT ##
    ################    
    indices_traintest = list(range(len(pydata.dat))) #Create list of indices
    test_indices = indices_traintest[::int(test_split*100)]  
    train_val_indices = [x for x in indices_traintest if x not in test_indices]

    val_indices = train_val_indices[::int(validation_split*100)]
    train_indices = [x for x in train_val_indices if x not in val_indices]
   
  
    traindata = [pydata[i] for i in train_indices]
    valdata = [pydata[i] for i in val_indices]
    testdata = [pydata[i] for i in test_indices]

    #Defining loaders
    testloader = torch.utils.data.DataLoader(testdata, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = True,
                                        drop_last= False,
                                        collate_fn = dataprep.BatchTransformation())

    validloader = torch.utils.data.DataLoader(valdata, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = True,
                                            drop_last= False,
                                            collate_fn = dataprep.BatchTransformation())

    trainloader = torch.utils.data.DataLoader(traindata, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = True,
                                            drop_last= False,
                                            collate_fn = dataprep.BatchTransformation())


    ###########################################################################################
    ################################## 2. Defining model ######################################
    ###########################################################################################
    #Defining network
    if modality == "PPG":
        model = networks.PPGNet(tensor_length = padding_length, drop = drop)
    elif modality == "GSR":
        model = networks.GSRNet(tensor_length = padding_length, drop = drop)
    elif modality == "EEG":
        model = networks.EEGNet(tensor_length = padding_length, drop = drop)
    
    print(model)

    #Loss function & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.0001)




    ###########################################################################################
    ########################## 3. Training & Validation loop ##################################
    ###########################################################################################
    
    if trainortest == "train":
        
        #Prerequisites
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

                torch.save(model.state_dict(), "pytorch/trained_models/"+participant+"_"+modality+".pt")
                valid_loss_min = valid_loss




        plt.plot(list(range(epochs-5)), train_list[5:len(train_list)], label = "train")
        plt.plot(list(range(epochs-5)), valid_list[5:len(valid_list)], label = "validation")
        plt.show()
        



    ###########################################################################################
    ########################## 4. Assessing model performance ##################################
    ###########################################################################################
    elif trainortest == "test":
        model.load_state_dict(torch.load("pytorch/trained_models/"+participant+"_"+modality+".pt"))

        
        model.eval()

        test_loss = 0

        diff = torch.Tensor()
        
        predictions = torch.Tensor()
        labelss = torch.Tensor()

        for windows, labels in testloader:
            
            #Test pass    
            out = model(windows)
            loss = criterion(out.squeeze(), labels)
            test_loss += loss.item()*windows.size(0)

            foo = (out.squeeze() - labels)
            diff = torch.cat([diff,foo])

            predictions = torch.cat([predictions, out])
            labelss = torch.cat([labelss, labels])

        test_loss = test_loss/len(testloader.sampler)
        print("Test los:",test_loss)
        average_miss = sum(abs(diff))/len(testloader.sampler)

        print("Average Missclasification:", float(average_miss))
        print("Or on the orignal scale:", float(average_miss*20))

        corr = np.corrcoef(predictions.squeeze().detach().numpy(), labelss.detach().numpy())
        print("Correlation predictions and labels:", float(corr[1][0]))

        print(predictions.squeeze()) 
        print(labelss.squeeze())            
        
        plt.hist(diff.detach().numpy(), bins= 50)
        plt.show()
        print('dd')
