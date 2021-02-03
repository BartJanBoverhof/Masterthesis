"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description Single-modular deep neural network design for the EEG-modality.
"""



################### 0. Prerequisites ###################
#Loading packages
import torch #PyTorch deep-learning library
from torch import nn, optim #PyTorch additionals and training optimizer
import torch.nn.functional as F #PyTorch library providing a lot of pre-specified functions



################### EEG Net ###################
class EegNet(nn.Module):
    def __init__(self):
        super(EegNet, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv1d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv1d(3, 16, 3, padding=1)

        #Max pooling layer (3x1)
        self.pool = nn.MaxPool1d(kernel_size = 3, stride = 1) 

        #Batch normalization
        self.batch1 = nn.BatchNorm1d(num_features = 1)
        self.batch2 = nn.BatchNorm1d(num_features = 1)
        self.batch3 = nn.BatchNorm1d(num_features = 1)
        self.batch4 = nn.BatchNorm1d(num_features = 1)

        #Dense layer
        self.dense = nn.Linear(100,10) 
        
    def forward(self, x): 
        x = self.pool(F.elu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.elu(self.batch2(self.conv2(x)))) #Second block
        x = self.pool(F.elu(self.batch3(self.conv3(x)))) #Third block
        x = self.pool(F.elu(self.batch4(self.conv4(x)))) #Fourth block
        x = F.softmax(self.dense(x)) #Classification block
        return x

#Display network
#eeg_net = EegNet()
#print(eeg_net)



################### PPG Net ###################
class PpgNet(nn.Module):
    def __init__(self):
        super(PpgNet, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv1d(3, 16, 3, padding=1)
        self.conv1 = nn.Conv1d(3, 16, 3, padding=1)

        #Max pooling layer (3x1)
        self.pool = nn.MaxPool1d(kernel_size = 4, stride = 1) 

        #Batch normalization
        self.batch1 = nn.BatchNorm1d(num_features = 1)
        self.batch2 = nn.BatchNorm1d(num_features = 1)

        #LSTM layers
        self.lstm = nn.LSTM(10)

        #Dense layer
        self.dense = nn.Linear(100,10) 
        
        
    def forward(self, x): 
        x = self.pool(F.relu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.relu(self.batch2(self.conv2(x)))) #Second block
        x = self.lstm(x) #Third block
        x = self.lstm(x) #Fourth block
        x = F.softmax(self.dense(x)) #Classification block
        return x

#Display network
#PpgNet()



################### GSR Net ###################
class GsrNet(nn.Module):
    def __init__(self):
        super(GsrNet, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv1d(3, 16, 3, padding=1)
        self.conv1 = nn.Conv1d(3, 16, 3, padding=1)

        #Max pooling layer (3x1)
        self.pool = nn.MaxPool1d(kernel_size = 4, stride = 1) 

        #Batch normalization
        self.batch1 = nn.BatchNorm1d(num_features = 1)
        self.batch2 = nn.BatchNorm1d(num_features = 1)

        #LSTM layers
        self.lstm = nn.LSTM(10)

        #Dense layer
        self.dense = nn.Linear(100,10) 
        
        
    def forward(self, x): 
        x = self.pool(F.relu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.relu(self.batch2(self.conv2(x)))) #Second block
        x = self.lstm(x) #Third block
        x = self.lstm(x) #Fourth block
        x = F.softmax(self.dense(x)) #Classification block
        return x

#Display network
#GsrNet()



################### Multi-modular Net ###################
foo = 1




################### Test Net ###################
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x

Network()