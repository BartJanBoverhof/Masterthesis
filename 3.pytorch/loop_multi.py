#!/usr/bin/env python
"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description This file contains the central training-loop function for training the multi-modular network.
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
    import utility, networks
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)


def MultiTrainLoop(participant, hpo, epochs, trainortest, batch_size):
    """
    Purpose:
        Central training loop function for the multi-modal network. 
    Arguments:
        participant: particpant to select for training.
        modality: network variation to train (EEG / PPG / GSR / Multi) 
        batch_size: utilized batch size for training
        hpo: objtect containing optimized hyperparamaters to utilize for training
        trainortest: whether to train, or test the already trained model at hand
    """
    ###########################################################################################
    ########################## 1. Create PyTorch dataset & Loader(s) ##########################
    ###########################################################################################
    #Create PyTorch dataset definition class
    path = "pipeline/prepared_data/"+participant+".pickle"
                                        
    eegdat =  utility.PytorchDataset(path = path,       #Creating PyTorch dataset
                                       modality = "EEG")
    ppgdat =  utility.PytorchDataset(path = path,       #Creating PyTorch dataset
                                      modality = "PPG")
    gsrdat =  utility.PytorchDataset(path = path,       #Creating PyTorch dataset
                                      modality = "GSR")

    print(participant)

    padinglength_eeg = utility.PaddingLength(eegdat) #Determining the longest window for later use
    padinglength_ppg = utility.PaddingLength(ppgdat) #Determining the longest window for later use
    padinglength_gsr = utility.PaddingLength(gsrdat) #Determining the longest window for later use

    utility.BatchTransformationEEG.transfer([padinglength_eeg, "EEG"]) #Transfer max padding length & modality vars to BatchTransfor class               
    utility.BatchTransformationPPG.transfer([padinglength_ppg, "PPG"]) #Transfer max padding length & modality vars to BatchTransfor class
    utility.BatchTransformationGSR.transfer([padinglength_gsr, "GSR"]) #Transfer max padding length & modality vars to BatchTransfor class
               
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
                                            collate_fn = utility.BatchTransformationEEG())

    eeg_validloader = torch.utils.data.DataLoader(eeg_val, #Validation loader
                                            batch_size = batch_size, 
                                            drop_last= False,
                                            shuffle = False,                                            
                                            collate_fn = utility.BatchTransformationEEG())

    eeg_testloader = torch.utils.data.DataLoader(eeg_test, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= False,
                                        collate_fn = utility.BatchTransformationEEG())



    ppg_trainloader = torch.utils.data.DataLoader(ppg_train, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = utility.BatchTransformationPPG())

    ppg_validloader = torch.utils.data.DataLoader(ppg_val, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = utility.BatchTransformationPPG())

    ppg_testloader = torch.utils.data.DataLoader(ppg_test, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= False,
                                        collate_fn = utility.BatchTransformationPPG())                                            

    gsr_trainloader = torch.utils.data.DataLoader(gsr_train, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = utility.BatchTransformationGSR())

    gsr_validloader = torch.utils.data.DataLoader(gsr_val, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = utility.BatchTransformationGSR())

    gsr_testloader = torch.utils.data.DataLoader(gsr_test, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= False,
                                        collate_fn = utility.BatchTransformationGSR())


    ###########################################################################################
    ################################## 2. Defining model ######################################
    ###########################################################################################
    #Defining network
    multi_model = networks.MULTINet(eegtensor_length = padinglength_eeg,
                                    ppgtensor_length = padinglength_ppg,
                                    gsrtensor_length = padinglength_gsr,
                                    drop = hpo[participant]["dropout_rate"],
                                    units_1 = hpo[participant]["dense1"],
                                    units_2 = hpo[participant]["dense2"])
    
    #Loss function & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(multi_model.parameters(), lr = hpo[participant]["lr"])




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



    ###########################################################################################
    ########################## 4. Assessing model performance ##################################
    ###########################################################################################
    elif trainortest == "test":
        multi_model.load_state_dict(torch.load("pytorch/trained_models/multi/"+participant+"_multi.pt", map_location= "cpu"))

        test_loss = 0.0

        
        multi_model.eval()
        diff = torch.Tensor()
        
        predictions_concat = torch.Tensor()
        labels_concat = torch.Tensor()

        for (eeg_windows, labels), (ppg_windows, labels), (gsr_windows, labels) in zip(eeg_testloader, ppg_testloader, gsr_testloader):
            
            #Test pass    
            out = multi_model(eeg_windows, ppg_windows, gsr_windows)
            loss = criterion(out.squeeze(), labels)
            test_loss += loss.item()*eeg_windows.size(0)

            foo = (out.squeeze() - labels)
            diff = torch.cat([diff,foo])

            predictions_concat = torch.cat([predictions_concat, out])
            labels_concat = torch.cat([labels_concat, labels])

    return predictions_concat, labels_concat

