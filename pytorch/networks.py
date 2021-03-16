"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description Single-modular deep neural network design for the EEG-modality.
"""



################### 0. Prerequisites ###################
#Loading packages
import torch #PyTorch deep-learning library
from torch import optim #PyTorch additionals and training optimizer
import torch.nn.functional as F #PyTorch library providing a lot of pre-specified functions
import torch.nn as nn
from torchsummary import summary

################### EEG Net ###################
class EEGNet(nn.Module):
    def __init__(self, tensor_length, drop = 0.25):
        super(EEGNet, self).__init__()

        self.drop = drop
        self.tensor_length = tensor_length
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

        #Dropout layer
        self.dropout = nn.Dropout(drop)

        
    def forward(self, x): 
        x = self.pool(F.elu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.elu(self.batch2(self.conv2(x)))) #Second block
        x = self.pool(F.elu(self.batch3(self.conv3(x)))) #Third block
        x = self.pool(F.elu(self.batch4(self.conv4(x)))) #Fourth block
        
        x = x.view(-1, x.shape[1]* x.shape[2]) #Flatten
        x = self.dropout(x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)

        return x

class PPGNet(nn.Module):
    def __init__(self, tensor_length, drop = 0.25):
        super(PPGNet, self).__init__()

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

        #Dropout layer
        self.dropout = nn.Dropout(drop)

        
    def forward(self, x): 
        x = self.pool(F.elu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.elu(self.batch2(self.conv2(x)))) #Second block
        x = self.pool(F.elu(self.batch3(self.conv3(x)))) #Third block
        x = self.pool(F.elu(self.batch4(self.conv4(x)))) #Fourth block
        
        x = x.view(-1, x.shape[1]* x.shape[2]) #Flatten
        x = self.dropout(x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)

        return x

class GSRNet(nn.Module):
    def __init__(self, tensor_length, drop = 0.25):
        super(GSRNet, self).__init__()

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

        #Dropout layer
        self.dropout = nn.Dropout(drop)

        
    def forward(self, x): 
        x = self.pool(F.elu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.elu(self.batch2(self.conv2(x)))) #Second block
        x = self.pool(F.elu(self.batch3(self.conv3(x)))) #Third block
        x = self.pool(F.elu(self.batch4(self.conv4(x)))) #Fourth block
        
        x = x.view(-1, x.shape[1]* x.shape[2]) #Flatten
        x = self.dropout(x)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)

        return x

################### Multi-modular Net ###################



################### Test Net ###################