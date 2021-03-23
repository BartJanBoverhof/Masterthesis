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
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ChainDataset
from torch import optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np

try: #Importing network
    import dataprep, networks
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)


def TrainLoop(participant, drop, epochs, trainortest, batch_size):
    
    ###########################################################################################
    ########################## 1. Create PyTorch dataset & Loader(s) ##########################
    ###########################################################################################
    #Create PyTorch dataset definition class
    path = "pipeline/prepared_data/"+participant+".pickle"
                                        
    eegdat =  dataprep.PytorchDataset(path = path,       #Creating PyTorch dataset
                                       modality = "EEG")
    ppgdat =  dataprep.PytorchDataset(path = path,       #Creating PyTorch dataset
                                      modality = "PPG")
    gsrdat =  dataprep.PytorchDataset(path = path,       #Creating PyTorch dataset
                                      modality = "GSR")

    same = len(eegdat) == len(ppgdat) == len(gsrdat)
    if same == True:
        print("Amount of windows are equal across modalities")
    else:
        print("BEWARE AMOUNT OF WINDOWS DIFFER ACROSS MODALITIES!!!")

    padinglength_eeg = dataprep.PaddingLength(eegdat) #Determining the longest window for later use
    padinglength_ppg = dataprep.PaddingLength(ppgdat) #Determining the longest window for later use
    padinglength_gsr = dataprep.PaddingLength(gsrdat) #Determining the longest window for later use

    dataprep.BatchTransformationEEG.transfer([padinglength_eeg, "EEG"]) #Transfer max padding length & modality vars to BatchTransfor class               
    dataprep.BatchTransformationPPG.transfer([padinglength_ppg, "PPG"]) #Transfer max padding length & modality vars to BatchTransfor class
    dataprep.BatchTransformationGSR.transfer([padinglength_gsr, "GSR"]) #Transfer max padding length & modality vars to BatchTransfor class
               
    #Making splits
    test_split = .1
    validation_split = .1

    ################
    ## TEST SPLIT ##
    ################    
    indices_traintest = list(range(len(eegdat.dat))) #Create list of indices
    test_indices = indices_traintest[::int(test_split*100)]  
    train_val_indices = [x for x in indices_traintest if x not in test_indices]

    val_indices = train_val_indices[::int(validation_split*100)]
    train_indices = [x for x in train_val_indices if x not in val_indices]
   
  
    eeg_train = [eegdat[i] for i in train_indices]
    eeg_val = [eegdat[i] for i in val_indices]
    eeg_test = [eegdat[i] for i in test_indices]

    ppg_train = [ppgdat[i] for i in train_indices]
    ppg_val = [ppgdat[i] for i in val_indices]
    ppg_test = [ppgdat[i] for i in test_indices]

    gsr_train = [gsrdat[i] for i in train_indices]
    gsr_val = [gsrdat[i] for i in val_indices]
    gsr_test = [gsrdat[i] for i in test_indices]

    #Defining loaders
    eeg_trainloader = torch.utils.data.DataLoader(eeg_train, #Training loader
                                            batch_size = batch_size, 
                                            drop_last= False,
                                            shuffle = False,
                                            collate_fn = dataprep.BatchTransformationEEG())

    eeg_validloader = torch.utils.data.DataLoader(eeg_val, #Validation loader
                                            batch_size = batch_size, 
                                            drop_last= False,
                                            shuffle = False,                                            
                                            collate_fn = dataprep.BatchTransformationEEG())

    eeg_testloader = torch.utils.data.DataLoader(eeg_test, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= False,
                                        collate_fn = dataprep.BatchTransformationEEG())



    ppg_trainloader = torch.utils.data.DataLoader(ppg_train, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = dataprep.BatchTransformationPPG())

    ppg_validloader = torch.utils.data.DataLoader(ppg_val, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = dataprep.BatchTransformationPPG())

    ppg_testloader = torch.utils.data.DataLoader(ppg_test, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= False,
                                        collate_fn = dataprep.BatchTransformationPPG())                                            

    gsr_trainloader = torch.utils.data.DataLoader(gsr_train, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = dataprep.BatchTransformationGSR())

    gsr_validloader = torch.utils.data.DataLoader(gsr_val, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = dataprep.BatchTransformationGSR())

    gsr_testloader = torch.utils.data.DataLoader(gsr_test, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= False,
                                        collate_fn = dataprep.BatchTransformationGSR())


    ###########################################################################################
    ################################## 2. Defining model ######################################
    ###########################################################################################
    #Defining network
    multi_model = networks.MULTINet(eegtensor_length = padinglength_eeg,
                                    ppgtensor_length = padinglength_ppg,
                                    gsrtensor_length = padinglength_gsr,
                                    drop = drop)
    
    #Loss function & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(multi_model.parameters(), lr= 0.0001)




    ###########################################################################################
    ########################## 3. Training & Validation loop ##################################
    ###########################################################################################
    
    if trainortest == "train":

        train_list = []
        valid_list = []
        valid_loss_min = np.Inf

        for epoch in range(1, epochs+1):
                
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            ###Training loop###
            ###################
            multi_model.train()
            for (eeg_windows, labels), (ppg_windows, labels), (gsr_windows, labels) in zip(eeg_trainloader, ppg_trainloader, gsr_trainloader):

                #Training pass
                optimizer.zero_grad()
                out = multi_model(eeg_windows, ppg_windows, gsr_windows)


                loss = criterion(out.squeeze(), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * eeg_windows.size(0)


            ###################
            ##Validation loop##
            ###################
            multi_model.eval()
            for (eeg_windows, labels), (ppg_windows, labels), (gsr_windows, labels) in zip(eeg_validloader, ppg_validloader, gsr_validloader):
                
                #Validation pass
                out = multi_model(eeg_windows, ppg_windows, gsr_windows)
                loss = criterion(out.squeeze(), labels)
                valid_loss += loss.item()*eeg_windows.size(0)

            #Averages losses
            train_loss = train_loss/len(eeg_trainloader.sampler)
            valid_loss = valid_loss/len(eeg_validloader.sampler)
            
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
            
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))

                torch.save(multi_model.state_dict(), "pytorch/trained_models/"+participant+".pt")
                valid_loss_min = valid_loss




        plt.plot(list(range(epochs-5)), train_list[5:len(train_list)], label = "train")
        plt.plot(list(range(epochs-5)), valid_list[5:len(valid_list)], label = "validation")
        plt.show()
    



    ###########################################################################################
    ########################## 4. Assessing model performance ##################################
    ###########################################################################################
    elif trainortest == "test":
        multi_model.load_state_dict(torch.load("pytorch/trained_models/"+participant+".pt"))

        test_loss = 0.0

        
        multi_model.eval()
        diff = torch.Tensor()
        
        predictions = torch.Tensor()
        labelss = torch.Tensor()

        for (eeg_windows, labels), (ppg_windows, labels), (gsr_windows, labels) in zip(eeg_validloader, ppg_validloader, gsr_validloader):
            
            #Test pass    
            out = multi_model(eeg_windows, ppg_windows, gsr_windows)
            loss = criterion(out.squeeze(), labels)
            test_loss += loss.item()*eeg_windows.size(0)

            foo = (out.squeeze() - labels)
            diff = torch.cat([diff,foo])

            predictions = torch.cat([predictions, out])
            labelss = torch.cat([labelss, labels])

        test_loss = test_loss/len(eeg_testloader.sampler)
        print("Test los:",test_loss)
        average_miss = sum(abs(diff))/len(eeg_testloader.sampler)

        print("Average Missclasification:", float(average_miss))
        print("Or on the orignal scale:", float(average_miss*20))

        corr = np.corrcoef(predictions.squeeze().detach().numpy(), labelss.detach().numpy())
        print("Correlation predictions and labels:", float(corr[1][0]))

        print(predictions.squeeze()) 
        print(labelss.squeeze())            
        
        plt.hist(diff.detach().numpy(), bins= 50)
        plt.show()

