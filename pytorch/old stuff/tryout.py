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
    """
    indices_traintest = list(range(len(pydata.dat))) #Create list of indices
    test_indices = indices_traintest[::int(test_split*100)]  
    train_val_indices = [x for x in indices_traintest if x not in test_indices]

    val_indices = train_val_indices[::int(validation_split*100)]
    train_indices = [x for x in train_val_indices if x not in val_indices]
   
  
    traindata = [pydata[i] for i in train_indices]
    valdata = [pydata[i] for i in val_indices]
    testdata = [pydata[i] for i in test_indices]
    """
    #TEMP
    indices_traintest = list(range(len(pydata.dat))) #Create list of indices
    cuttoff = int(len(indices_traintest)*test_split)
    test_indices = indices_traintest[-cuttoff:]
    train_indices = indices_traintest[:len(pydata)-cuttoff]

    traindata = [pydata[i] for i in train_indices]
    testdata = [pydata[i] for i in test_indices]
     
    testloader = torch.utils.data.DataLoader(testdata, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= True,
                                        collate_fn = dataprep.BatchTransformation())

    trainloader = torch.utils.data.DataLoader(traindata, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= True,
                                            collate_fn = dataprep.BatchTransformation())

    """
    #Defining loaders
    testloader = torch.utils.data.DataLoader(testdata, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = True,
                                        drop_last= True,
                                        collate_fn = dataprep.BatchTransformation())

    validloader = torch.utils.data.DataLoader(valdata, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = True,
                                            drop_last= True,
                                            collate_fn = dataprep.BatchTransformation())

    trainloader = torch.utils.data.DataLoader(traindata, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = True,
                                            drop_last= True,
                                            collate_fn = dataprep.BatchTransformation())
    """


    ###########################################################################################
    ################################## 2. Defining model ######################################
    ###########################################################################################
    #Defining network
    if modality == "PPG":
        model = networks.PPGNet(drop = drop)
    elif modality == "GSR":
        model = networks.GSRNet(drop = drop)
    elif modality == "PPG":
        model = networks.EEGNet(tensor_length = padding_length, drop = drop)
    
    print(model)

    #Loss function & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.0001)







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
                
                #Averages losses
                train_loss = train_loss/len(trainloader.sampler)
                

                print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss:'.format(
                epoch, train_loss))

                torch.save(model.state_dict(), "pytorch/trained_models/"+participant+"_"+modality+".pt")




    ###########################################################################################
    ########################## 4. Assessing model performance ##################################
    ###########################################################################################
    elif trainortest == "test":
        model.load_state_dict(torch.load("pytorch/trained_models/"+participant+"_"+modality+".pt"))

        test_loss = 0.0

        if modality == "GSR" or modality == "PPG":
            h = model.InitHiddenstate(batch_size)

            model.eval()
            diff = torch.Tensor()
            
            predictions = torch.Tensor()
            labelss = torch.Tensor()

            for windows, labels in testloader:
                
                #Test pass    
                h = tuple([each.data for each in h])
                out, h = model(windows, h)
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

        elif modality == "EEG":
            
            model.eval()
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
