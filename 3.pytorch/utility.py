"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description This file contains all additional utility function required throughout the training and testing process
"""

#Loading packages
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pickle
import torch



class PytorchDataset(Dataset):
    """
    Purpose:
        Class that combines all functions to create PyTorch dataset and sample from it. 
    Arguments:
        participant: particpant to select for training.
        modality: network variation to train (EEG / PPG / GSR / Multi) 
        batch_size: utilized batch size for training
        hpo: objtect containing optimized hyperparamaters to utilize for training
        trainortest: whether to train, or test the already trained model at hand
    """

    def __init__(self, path, modality):
        """
        Purpose: 
            Load pickle object and save only required data. 
        """
        dat = pickle.load(open(path, "rb")) #Open pickle
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
            Return the length of the dataset.
        """
        return len(self.dat)


    def __getitem__(self, idx):
        """
        Purpose:
            Iterater through dataset to select windows.
        """
        windows = self.dat[idx]
        labels = self.labels[idx]

        return windows, labels
    

    def __ObtainModality__(self):
        """
        Purpose:
            Return current modality.
        """
        
        return self.modality



class BatchTransformation():
    def __call__(self, batch):
        """
        Purpose:   
            Class combining general batch transformation functions utilized to prepare 
            a batch and its labels before it is being fed into the network.
        Return:
            Properly transformed data and labels batch. 
        """

        padding_length = self.padding_length
        modality = self.modality

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
        BatchTransformation.padding_length = self[0]
        BatchTransformation.modality = self[1]


        
class BatchTransformationEEG():
    def __call__(self, batch):
        """
        Purpose:   
            Class combining general batch transformation functions utilized to prepare 
            a batch and its labels before it is being fed into the network.
            Only utilized for the EEG data when training / testing the multi-modular network.
        Return:
            Properly transformed data and labels batch. 
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
            Class combining general batch transformation functions utilized to prepare 
            a batch and its labels before it is being fed into the network.
            Only utilized for the PPG data when training / testing the multi-modular network.
        Return:
            Properly transformed data and labels batch. 
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
            Class combining general batch transformation functions utilized to prepare 
            a batch and its labels before it is being fed into the network.
            Only utilized for the GSR data when training / testing the multi-modular network.
        Return:
            Properly transformed data and labels batch. 
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