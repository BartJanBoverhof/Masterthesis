"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description This file conducts the hyper-paramater optimization for multi-modular networks on GPU.
BEWARE! Running this code on a machine with NVIDIA GPU is highly recommended. 
"""

import numpy as np
import torch
import os
import torch 
from torch import optim #PyTorch additionals and training optimizer
import torch.nn as nn
import torch.nn.functional as F #PyTorch library providing a lot of pre-specified functions
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import optuna 
import pickle5 as pickle
import math

#############################################################################################
##################################### Data preperation ######################################
#############################################################################################
class PytorchDataset(Dataset):
    
    def __init__(self, participant, modality):
        """
        Purpose: 
            Load pickle object and save only specified data. 
        """
        dat = pickle.load(open("...SPECIFY THE LOCATION OF THE DATA HERE..."+participant+".pickle", "rb")) #Open pickle
        key = "labels_"+modality #Determining right dict key
        self.labels = dat[key] #Saving labels
        self.dat = dat[modality] #Saving only modality of interest
        self.modality = modality

        #Determining the longest window for later use
        lengths = []
        for i in self.dat:
            lengths.append(len(i))
        longest_window = max(lengths)


    def __len__(self):
        """
        Purpose: 
            Obtain the length of the data
        """
        return len(self.dat)


    def __getitem__(self, idx):
        """
        Purpose:
            Iterater to select windows
        """
        windows = self.dat[idx]
        labels = self.labels[idx]

        return windows, labels
    

    def __ObtainModality__(self):
        """
        Purpose:
            Print modality
        """
        
        return self.modality



class BatchTransformationEEG():
    def __call__(self, batch):
        """
        Purpose:   
            Transformation of windows per batch (padding & normalizing labels & transposing)
        """

        padding_length = self.padding_length

        #PADDING
        sequences = [x[0] for x in batch] #Get ordered windows
        sequences.append(torch.ones(padding_length,1)) #Temporary add window of desired padding length
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value = 0) #Pad
        sequences_padded = sequences_padded[0:len(batch)] #Remove the added window

        #Obtaining Sorted labels and standardizing
        labels = torch.tensor([x[1] for x in batch]) #Get ordered windows
        labels = (labels - 1)/ 20
        
        #TRANSPOSE BATCH 
        sequences_padded = torch.transpose(sequences_padded, 1, 2)

        return sequences_padded, labels

    def transfer(self):
        """
        Purpose:
            Transfering the earlier obtained padding length to the BatchTransformation class such it can be used
            in the __call__ function.
        """
        BatchTransformationEEG.padding_length = self[0]



class BatchTransformationPPG():
    def __call__(self, batch):
        """
        Purpose:   
            Transformation of windows per batch (padding & normalizing labels & transposing)
        """

        padding_length = self.padding_length

        #PADDING
        sequences = [x[0] for x in batch] #Get ordered windows
        sequences.append(torch.ones(padding_length,1)) #Temporary add window of desired padding length
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value = 0) #Pad
        sequences_padded = sequences_padded[0:len(batch)] #Remove the added window

        #Obtaining Sorted labels and standardizing
        labels = torch.tensor([x[1] for x in batch]) #Get ordered windows
        labels = (labels - 1)/ 20
        
        #TRANSPOSE BATCH 
        sequences_padded = torch.transpose(sequences_padded, 1, 2)

        return sequences_padded, labels

    def transfer(self):
        """
        Purpose:
            Transfering the earlier obtained padding length to the BatchTransformation class such it can be used
            in the __call__ function.
        """
        BatchTransformationPPG.padding_length = self[0]



class BatchTransformationGSR():
    def __call__(self, batch):
        """
        Purpose:   
            Transformation of windows per batch (padding & normalizing labels & transposing)
        """

        padding_length = self.padding_length

        #PADDING
        sequences = [x[0] for x in batch] #Get ordered windows
        sequences.append(torch.ones(padding_length,1)) #Temporary add window of desired padding length
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value = 0) #Pad
        sequences_padded = sequences_padded[0:len(batch)] #Remove the added window

        #Obtaining Sorted labels and standardizing
        labels = torch.tensor([x[1] for x in batch]) #Get ordered windows
        labels = (labels - 1)/ 20
        
        #TRANSPOSE BATCH 
        sequences_padded = torch.transpose(sequences_padded, 1, 2)

        return sequences_padded, labels


    def transfer(self):
        """
        Purpose:
            Transfering the earlier obtained padding length to the BatchTransformation class such it can be used
            in the __call__ function.
        """
        BatchTransformationGSR.padding_length = self[0]


def PaddingLength(data):
    """
    Purpose:
        Determine padding length
    """
    lengths = []
    for i in data.dat:
        lengths.append(i.shape[0])
    
    return max(lengths)


#############################################################################################
######################################## Networks ###########################################
#############################################################################################
class EEGNet(nn.Module):
    def __init__(self, tensor_length, drop , multi):
        super(EEGNet, self).__init__()

        self.multi = multi
        self.drop = drop

        foo = int(tensor_length /3)
        foo = int(foo /3)         
        foo = int(foo /3)
        foo = int(foo /3)
        dense_input = 200*foo

        #Convolutional layers
        self.conv1 = nn.Conv1d(in_channels = 4, out_channels = 25, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv1d(in_channels = 25, out_channels = 50, kernel_size = 3, padding=1)
        self.conv3 = nn.Conv1d(in_channels = 50, out_channels = 100, kernel_size = 3, padding=1)
        self.conv4 = nn.Conv1d(in_channels = 100, out_channels = 200, kernel_size = 3, padding=1)

        #Max pooling layer (3x1)
        self.pool = nn.MaxPool1d(kernel_size = 3, stride = 3) 

        #Batch normalization
        self.batch1 = nn.BatchNorm1d(num_features = 25)
        self.batch2 = nn.BatchNorm1d(num_features = 50)
        self.batch3 = nn.BatchNorm1d(num_features = 100)
        self.batch4 = nn.BatchNorm1d(num_features = 200)

        #Dense layer
        self.dense1 = nn.Linear(dense_input, int(dense_input/8)) 
        self.dense2 = nn.Linear(int(dense_input/8), 1) 
        self.dense3 = nn.Linear(dense_input, dense_input) 

        #Dropout layer
        self.dropout = nn.Dropout(drop)

        
    def forward(self, x): 
        x = self.pool(F.elu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.elu(self.batch2(self.conv2(x)))) #Second block
        x = self.pool(F.elu(self.batch3(self.conv3(x)))) #Third block
        x = self.pool(F.elu(self.batch4(self.conv4(x)))) #Fourth block
        
        x = x.view(-1, x.shape[1]* x.shape[2]) #Flatten
        
        if self.multi == False:
            x = self.dropout(x)
            x = F.relu(self.dense1(x))
            x = self.dense2(x)
        elif self.multi == True:
            x = F.relu(self.dense3(x))

        return x

class PPGNet(nn.Module):
    def __init__(self, tensor_length, drop, multi):
        super(PPGNet, self).__init__()

        self.multi = multi
        self.drop = drop
        self.tensor_length = tensor_length
        foo = int(tensor_length /3)
        foo = int(foo /3)         
        foo = int(foo /3)
        foo = int(foo /3)
        dense_input = 128*foo

        #Convolutional layers
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 3, padding=1)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding=1)
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, padding=1)

        #Max pooling layer (3x1)
        self.pool = nn.MaxPool1d(kernel_size = 3, stride = 3) 

        #Batch normalization
        self.batch1 = nn.BatchNorm1d(num_features = 16)
        self.batch2 = nn.BatchNorm1d(num_features = 32)
        self.batch3 = nn.BatchNorm1d(num_features = 64)
        self.batch4 = nn.BatchNorm1d(num_features = 128)

        #Dense layer
        self.dense1 = nn.Linear(dense_input, int(dense_input/8)) 
        self.dense2 = nn.Linear(int(dense_input/8), 1) 
        self.dense3 = nn.Linear(dense_input, dense_input) 

        #Dropout layer
        self.dropout = nn.Dropout(drop)

        
    def forward(self, x): 
        x = self.pool(F.elu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.elu(self.batch2(self.conv2(x)))) #Second block
        x = self.pool(F.elu(self.batch3(self.conv3(x)))) #Third block
        x = self.pool(F.elu(self.batch4(self.conv4(x)))) #Fourth block
        
        x = x.view(-1, x.shape[1]* x.shape[2]) #Flatten
        
        if self.multi == False:
            x = self.dropout(x)
            x = F.relu(self.dense1(x))
            x = self.dense2(x)
        elif self.multi == True:
            x = F.relu(self.dense3(x))

        return x

class GSRNet(nn.Module):
    def __init__(self, tensor_length, drop, multi):
        super(GSRNet, self).__init__()

        self.multi = multi
        self.drop = drop
        self.tensor_length = tensor_length
        foo = int(tensor_length /3)
        foo = int(foo /3)         
        foo = int(foo /3)
        foo = int(foo /3)
        dense_input = 128*foo

        #Convolutional layers
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 3, padding=1)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding=1)
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, padding=1)

        #Max pooling layer (3x1)
        self.pool = nn.MaxPool1d(kernel_size = 3, stride = 3) 

        #Batch normalization
        self.batch1 = nn.BatchNorm1d(num_features = 16)
        self.batch2 = nn.BatchNorm1d(num_features = 32)
        self.batch3 = nn.BatchNorm1d(num_features = 64)
        self.batch4 = nn.BatchNorm1d(num_features = 128)

        #Dense layer
        self.dense1 = nn.Linear(dense_input, int(dense_input/8)) 
        self.dense2 = nn.Linear(int(dense_input/8), 1) 
        self.dense3 = nn.Linear(dense_input, dense_input) 

        #Dropout layer
        self.dropout = nn.Dropout(drop)

        
    def forward(self, x): 
        x = self.pool(F.elu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.elu(self.batch2(self.conv2(x)))) #Second block
        x = self.pool(F.elu(self.batch3(self.conv3(x)))) #Third block
        x = self.pool(F.elu(self.batch4(self.conv4(x)))) #Fourth block
        
        x = x.view(-1, x.shape[1]* x.shape[2]) #Flatten

        if self.multi == False:
            x = self.dropout(x)
            x = F.relu(self.dense1(x))
            x = self.dense2(x)
        elif self.multi == True:
            x = F.relu(self.dense3(x))

        return x

################### Multi-modular Net ###################
class MULTINet(nn.Module):
    def __init__(self, eegtensor_length, ppgtensor_length, gsrtensor_length, drop, out_features):
        super(MULTINet, self).__init__()

        self.drop = drop
        
        egg_length = int(eegtensor_length /3)
        egg_length = int(egg_length /3)         
        egg_length = int(egg_length /3)
        egg_length = int(egg_length /3)
        egg_length = 200*egg_length

        ppg_length = int(ppgtensor_length /3)
        ppg_length = int(ppg_length /3)         
        ppg_length = int(ppg_length /3)
        ppg_length = int(ppg_length /3)
        ppg_length = 128*ppg_length

        gsr_length = int(gsrtensor_length /3)
        gsr_length = int(gsr_length /3)         
        gsr_length = int(gsr_length /3)
        gsr_length = int(gsr_length /3)
        gsr_length = 128*gsr_length

        concat = gsr_length+ppg_length+egg_length

        concat_final = math.ceil(concat/20)
        concat_final = math.ceil(concat_final/20)
        concat_final = 50*concat_final

        #Modality specific networks
        self.eegpart = EEGNet(drop = 0.25, tensor_length = eegtensor_length, multi = True)
        self.ppgpart = PPGNet(drop = 0.25, tensor_length = ppgtensor_length, multi = True)
        self.gsrpart = GSRNet(drop = 0.25, tensor_length = gsrtensor_length, multi = True)
        
        #Convolutional layers
        self.convhead1 = nn.Conv1d(in_channels = 1, out_channels = 25, kernel_size = 3, padding=1)
        self.convhead2 = nn.Conv1d(in_channels = 25, out_channels = 50, kernel_size = 3, padding=1)

        #Pooling layer
        self.poolhead = nn.MaxPool1d(kernel_size = 3, stride = 20) 

        #Batch normalization
        self.batchhead1 = nn.BatchNorm1d(num_features = 25)
        self.batchhead2 = nn.BatchNorm1d(num_features = 50)

        #Dropout
        self.dropouthead = nn.Dropout(drop)

        #Dense layers
        self.densehead2 = nn.Linear(concat_final, out_features) 
        self.densehead3 = nn.Linear(out_features, 1) 

    def forward(self, eeg_windows, ppg_windows, gsr_windows): 
        x = self.eegpart(eeg_windows)
        y = self.ppgpart(ppg_windows)
        z = self.gsrpart(gsr_windows)

        out = torch.cat([x,y,z],dim=1)
        out = out.unsqueeze(1)
        out = self.dropouthead(out)

        out = self.poolhead(F.relu(self.batchhead1(self.convhead1(out))))
        out = self.poolhead(F.relu(self.batchhead2(self.convhead2(out))))

        out = out.view(-1, out.shape[1]* out.shape[2]) #Flatten
        out = self.dropouthead(out)
        out = F.relu(self.densehead2(out))
        out = self.densehead3(out)

        return out

###########################################################################################
######################################### HPO search ######################################
###########################################################################################
def objective(trial):
    
    ###########################################################################################
    ########################## 1. Create PyTorch dataset & Loader(s) ##########################
    ###########################################################################################
    
    #Create PyTorch dataset definition class                               
    eegdat =  PytorchDataset(participant = participant,       #Creating PyTorch dataset
                                       modality = "EEG")
    ppgdat =  PytorchDataset(participant = participant,       #Creating PyTorch dataset
                                      modality = "PPG")
    gsrdat =  PytorchDataset(participant = participant,       #Creating PyTorch dataset
                                      modality = "GSR")

    same = len(eegdat) == len(ppgdat) == len(gsrdat)
    if same == True:
        print("Amount of windows are equal across modalities")
    else:
        print("BEWARE AMOUNT OF WINDOWS DIFFER ACROSS MODALITIES!!!")

    padinglength_eeg = PaddingLength(eegdat) #Determining the longest window for later use
    padinglength_ppg = PaddingLength(ppgdat) #Determining the longest window for later use
    padinglength_gsr = PaddingLength(gsrdat) #Determining the longest window for later use

    BatchTransformationEEG.transfer([padinglength_eeg, "EEG"]) #Transfer max padding length & modality vars to BatchTransfor class               
    BatchTransformationPPG.transfer([padinglength_ppg, "PPG"]) #Transfer max padding length & modality vars to BatchTransfor class
    BatchTransformationGSR.transfer([padinglength_gsr, "GSR"]) #Transfer max padding length & modality vars to BatchTransfor class
               
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
                                            collate_fn = BatchTransformationEEG())

    eeg_validloader = torch.utils.data.DataLoader(eeg_val, #Validation loader
                                            batch_size = batch_size, 
                                            drop_last= False,
                                            shuffle = False,                                            
                                            collate_fn = BatchTransformationEEG())

    eeg_testloader = torch.utils.data.DataLoader(eeg_test, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= False,
                                        collate_fn = BatchTransformationEEG())



    ppg_trainloader = torch.utils.data.DataLoader(ppg_train, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = BatchTransformationPPG())

    ppg_validloader = torch.utils.data.DataLoader(ppg_val, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = BatchTransformationPPG())

    ppg_testloader = torch.utils.data.DataLoader(ppg_test, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= False,
                                        collate_fn = BatchTransformationPPG())                                            

    gsr_trainloader = torch.utils.data.DataLoader(gsr_train, #Training loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = BatchTransformationGSR())

    gsr_validloader = torch.utils.data.DataLoader(gsr_val, #Validation loader
                                            batch_size = batch_size, 
                                            shuffle = False,
                                            drop_last= False,
                                            collate_fn = BatchTransformationGSR())

    gsr_testloader = torch.utils.data.DataLoader(gsr_test, #Test loader 
                                        batch_size = batch_size, 
                                        shuffle = False,
                                        drop_last= False,
                                        collate_fn = BatchTransformationGSR())


    ###########################################################################################
    ################################## 2. Defining model ######################################
    ###########################################################################################
    egg_length = int(padinglength_eeg /3)
    egg_length = int(egg_length /3)         
    egg_length = int(egg_length /3)
    egg_length = int(egg_length /3)
    egg_length = 200*egg_length

    ppg_length = int(padinglength_ppg /3)
    ppg_length = int(ppg_length /3)         
    ppg_length = int(ppg_length /3)
    ppg_length = int(ppg_length /3)
    ppg_length = 128*ppg_length

    gsr_length = int(padinglength_gsr /3)
    gsr_length = int(gsr_length /3)         
    gsr_length = int(gsr_length /3)
    gsr_length = int(gsr_length /3)
    gsr_length = 128*gsr_length

    concat = gsr_length+ppg_length+egg_length

    concat_final = round(concat/20)
    concat_final = round(concat_final/20)
    concat_final = 50*concat_final

    out_features = trial.suggest_int("out_units", 10, concat_final)
    drop = trial.suggest_float('dropout_rate', 0.2, 0.5)
    
    
    #Defining network
    multi_model = MULTINet(eegtensor_length = padinglength_eeg,
                                    ppgtensor_length = padinglength_ppg,
                                    gsrtensor_length = padinglength_gsr,
                                    drop = drop,
                                    out_features = out_features)
                                    
    
    #Loss function & Optimizer
    criterion = nn.MSELoss()
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log = True)
    optimizer = optim.Adam(multi_model.parameters(), lr= lr)

    if train_on_gpu:
        multi_model.cuda()

    ###########################################################################################
    ########################## 3. Training & Validation loop ##################################
    ###########################################################################################



    valid_loss_min = np.Inf

    for epoch in range(1, epochs+1):
            
        valid_loss = 0.0

        ###################
        ###Training loop###
        ###################
        multi_model.train()
        for (eeg_windows, labels), (ppg_windows, labels), (gsr_windows, labels) in zip(eeg_trainloader, ppg_trainloader, gsr_trainloader):

            if train_on_gpu:
                eeg_windows, ppg_windows, gsr_windows, labels = eeg_windows.cuda(), ppg_windows.cuda(), gsr_windows.cuda(), labels.cuda()

            #Training pass
            optimizer.zero_grad()
            out = multi_model(eeg_windows, ppg_windows, gsr_windows)


            loss = criterion(out.squeeze(), labels)
            loss.backward()
            optimizer.step()


        ###################
        ##Validation loop##
        ###################
        multi_model.eval()
        for (eeg_windows, labels), (ppg_windows, labels), (gsr_windows, labels) in zip(eeg_validloader, ppg_validloader, gsr_validloader):

            if train_on_gpu:
                eeg_windows, ppg_windows, gsr_windows, labels = eeg_windows.cuda(), ppg_windows.cuda(), gsr_windows.cuda(), labels.cuda()

            #Validation pass
            out = multi_model(eeg_windows, ppg_windows, gsr_windows)
            loss = criterion(out.squeeze(), labels)
            valid_loss += loss.item()*eeg_windows.size(0)

        #Averages losses
        valid_loss = valid_loss/len(eeg_test)
        
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:

            torch.save(multi_model.state_dict(), "model.pt")
            valid_loss_min = valid_loss




    with torch.no_grad():


        multi_model.load_state_dict(torch.load("model.pt"))

        diff = torch.FloatTensor()
        predictions = torch.FloatTensor()
        labelss = torch.FloatTensor()

        multi_model.eval()
        for (eeg_windows, labels), (ppg_windows, labels), (gsr_windows, labels) in zip(eeg_testloader, ppg_testloader, gsr_testloader):

            #Test pass    
            out = multi_model(eeg_windows, ppg_windows, gsr_windows)
            loss = criterion(out.squeeze(), labels)

            foo = (out.squeeze() - labels)
            diff = torch.cat([diff,foo])

        average_miss = sum(abs(diff))/len(eeg_test)

        accuracy = average_miss
        
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

###########################################################################################
############################################ Run ##########################################
###########################################################################################
participants = ["bci10", "bci12", "bci13", "bci17", "bci21", "bci22",
                "bci23", "bci24", "bci26", "bci27", "bci28", "bci29", "bci30", 
                "bci31", "bci32", "bci33", "bci34", "bci35", "bci36", "bci37", 
                "bci38", "bci39", "bci40", "bci41", "bci42", "bci43", "bci44"]



ntrials = 20
epochs = 40
batch_size = 10
np.random.seed(3791)
torch.manual_seed(3791)
torch.cuda.manual_seed(3791)

train_on_gpu = torch.cuda.is_available()



if __name__ == "__main__":
    for participant in participants:
        print(participant)
        torch.cuda.empty_cache()

        study = optuna.create_study(direction="minimize", 
                                    storage="sqlite:///example.db")
        study.optimize(objective, n_trials=ntrials)

        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

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

        with open("...SPECIFY YOUR HPO SAVE LOCATION HERE..."+participant+".pickle", 'wb') as handle: #Save as pickle
            pickle.dump(study.best_trial, handle, protocol=pickle.HIGHEST_PROTOCOL)