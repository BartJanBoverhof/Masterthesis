"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description Single-modular deep neural network design for the GSR-modality.
"""



################### 0. Prerequisites ###################
#Loading packages
import torch #PyTorch deep-learning library
from torch import nn, optim #PyTorch additionals and training optimizer
import torch.nn.functional as F #PyTorch library providing a lot of pre-specified functions



################### 1. Define Network architecture ###################
class GsrNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv1d(784, 256)
        self.conv2 = nn.Conv1d(784, 256)

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
GsrNet()