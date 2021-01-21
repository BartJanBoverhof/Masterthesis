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



################### 1. Define Network architecture ###################
class EegNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Convolutional layers
        self.conv1 = nn.Conv1d(784, 256)
        self.conv2 = nn.Conv1d(784, 256)
        self.conv3 = nn.Conv1d(784, 256)
        self.conv4 = nn.Conv1d(784, 256)

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
EegNet()


