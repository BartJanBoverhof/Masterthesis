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
import optuna

try: #Importing network
    import dataprep, networks
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)



participants = ["bci10", "bci12", "bci13", "bci17", "bci20", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", 
                "bci38", "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]

drop = 0.25
epochs = 5
trainortest = "train"
participant = participants[4]
modality = "EEG"

np.random.seed(3791)
torch.manual_seed(3791)

def objective(trial):    
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
    batch_size = 10
    validation_split = .1

    ################
    ## TEST SPLIT ##
    ################    
    indices = list(range(len(pydata.dat))) #Create list of indices
    np.random.shuffle(indices) #Shuffle indices.

    train_valid_split = int(np.floor(validation_split * len(indices))) #Calculate number of windows to use for val/train
    train_indices, val_indices = indices[train_valid_split:], indices[:train_valid_split] #Splitted indices.

    #Defining samplers
    train_sampler = SubsetRandomSampler(train_indices) #Train sampler
    valid_sampler = SubsetRandomSampler(val_indices) #Validation sampler
    
    #Defining loaders
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
        model = networks.PPGNet(drop = drop)
    elif modality == "GSR":
        model = networks.GSRNet(drop = drop)
    elif modality == "EEG":
        model = networks.EEGNet(tensor_length = padding_length, drop = drop)
    
    #Hyperparams
    #Optimizer and learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log = True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()
  



    ###########################################################################################
    ########################## 3. Training & Validation loop ##################################
    ###########################################################################################
    
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

                    torch.save(model.state_dict(), "pytorch/trained_models/"+participant+"_"+modality+".pt")
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
            diff = torch.Tensor()
            
            predictions = torch.Tensor()
            labelss = torch.Tensor()

            for windows, labels in validloader:
                
                #Test pass    
                out = model(windows)
                loss = criterion(out.squeeze(), labels)
                valid_loss += loss.item()*windows.size(0)

                foo = (out.squeeze() - labels)
                diff = torch.cat([diff,foo])

                predictions = torch.cat([predictions, out])
                labelss = torch.cat([labelss, labels])

            average_miss = sum(abs(diff))/len(validloader.sampler)
            """
            print("Epoch"+epoch+"\tAverage_miss="+average_miss)
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))

                torch.save(model.state_dict(), "pytorch/trained_models/"+participant+"_"+modality+".pt")
                valid_loss_min = valid_loss
            """

            accuracy = average_miss
            
            trial.report(accuracy, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return accuracy
    

if __name__ == "__main__":
    
    study_name = participant+"_hpo"
    folder = "HPO/search"

    study = optuna.create_study(direction="minimize", study_name= study_name, storage="sqlite:///example.db")
    study.optimize(objective, n_trials=5, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
