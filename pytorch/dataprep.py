from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pickle
import torch


class PytorchDataset(Dataset):
    
    def __init__(self, path, modality):
        """
        Purpose: 
            Load pickle object and save only specified data. 
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
    

    def __PaddingLength__(self):
        """
        Purpose:
            Determine padding length
        """
        lengths = []
        for i in self.dat:
            lengths.append(i.shape[0])
        
        return max(lengths)

    def __ObtainModality__(self):
        """
        Purpose:
            Print modality
        """
        
        return self.modality


#Batch transformation class
class BatchTransformation():
    def __call__(self, batch):
        """
        Purpose:   
            Transformation of windows per batch (padding & normalizing labels & transposing)
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


