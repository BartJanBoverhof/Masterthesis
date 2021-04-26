"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description This file contains the training-loop function for training the single-modular networks.
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
    import utility, networks
except ModuleNotFoundError:
    wd = os.getcwd()
    print("Error: please make sure that working directory is set as '~/Masterthesis'")
    print("Current working directory is:", wd)


def SingleTrainLoop(participant, modality, batch_size, hpo, epochs, trainortest):
    """
    Purpose:
        Central training loop function for the single-modulair network. 
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
    pydata =  utility.PytorchDataset(path = path,       #Creating PyTorch dataset
                                      modality = modality)

    padding_length = utility.PaddingLength(pydata) #Determining the longest window for later use
    utility.BatchTransformation.transfer([padding_length, modality]) #Transfer max padding length & modality vars to BatchTransfor class
               
               
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
                                        collate_fn = utility.BatchTransformation())

    validloader = torch.utils.data.DataLoader(valdata, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = True,
                                            drop_last= False,
                                            collate_fn = utility.BatchTransformation())

    trainloader = torch.utils.data.DataLoader(traindata, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = True,
                                            drop_last= False,
                                            collate_fn = utility.BatchTransformation())


    ###########################################################################################
    ################################## 2. Defining model ######################################
    ###########################################################################################
    #Defining network
    if modality == "PPG":
        model = networks.PPGNet(tensor_length = padding_length, 
                                drop = hpo[participant]["dropout_rate"], 
                                n_units = hpo[participant]["n_units"], 
                                multi = False)
    elif modality == "GSR":
        model = networks.GSRNet(tensor_length = padding_length, 
                                drop = hpo[participant]["dropout_rate"],
                                n_units = hpo[participant]["n_units"], 
                                multi = False)        
    elif modality == "EEG":
        model = networks.EEGNet(tensor_length = padding_length, 
                                drop = hpo[participant]["dropout_rate"],
                                n_units = hpo[participant]["n_units"], 
                                multi = False)   
    
    #Loss function & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr= hpo[participant]["lr"])




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

                torch.save(model.state_dict(), "pytorch/trained_models/"+modality+"/"+participant+"_"+modality+".pt")
                valid_loss_min = valid_loss


    ###########################################################################################
    #################################### 4. Test loop #########################################
    ###########################################################################################
    elif trainortest == "test":
        model.load_state_dict(torch.load("pytorch/trained_models/"+modality+"/"+participant+"_"+modality+".pt", map_location = "cpu"))

        
        model.eval()

        test_loss = 0

        diff = torch.Tensor()
        
        predictions_concat = torch.Tensor()
        labels_concat = torch.Tensor()

        for windows, labels in testloader:
            
            #Test pass    
            out = model(windows)
            loss = criterion(out.squeeze(), labels)
            test_loss += loss.item()*windows.size(0)

            foo = (out.squeeze() - labels)
            diff = torch.cat([diff,foo])

            predictions_concat = torch.cat([predictions_concat, out])
            labels_concat = torch.cat([labels_concat, labels])

    return predictions_concat, labels_concat