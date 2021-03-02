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

#Display network
#eeg_net = EegNet()
#summary(eeg_net, (1000,1))

################### PPG Net ###################
class PPGNet(nn.Module):

    def __init__(self, filters = 32, hidden_dim = 64, n_layers =2, drop = 0.25):
        super(PPGNet, self).__init__()
        
        self.ip_dim = filters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop = drop

        #Convolutional layers
        self.conv1 = nn.Conv1d(in_channels =  1, out_channels = filters, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv1d(in_channels = filters, out_channels = filters, kernel_size = 3, padding=1)

        #Max pooling layer (4)
        self.pool = nn.MaxPool1d(kernel_size = 4, stride = 4) 

        #Batch normalization
        self.batch1 = nn.BatchNorm1d(num_features = filters)
        self.batch2 = nn.BatchNorm1d(num_features = filters)
        self.batch3 = nn.BatchNorm1d(num_features = int(hidden_dim/4))

        #Dense & classification layer 
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim/4))
        self.fc2 = nn.Linear(int(hidden_dim/4), 1)


        #LSTM layer
        self.lstm = nn.LSTM(input_size = filters, 
                                hidden_size = hidden_dim, 
                                num_layers = n_layers, 
                                dropout = drop,
                                batch_first= True)
    
    def InitHiddenstate(self, batch_size):
           
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

    def forward(self, x, hidden): 

        batch_size = x.shape[0]
        #Convolutional block
        x = self.pool(F.relu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.relu(self.batch2(self.conv2(x)))) #Second block
        
        #LSTM-block
        x = torch.transpose(x, 1, 2)
        out, hidden = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)

        #Prediction block
        out = F.relu(self.batch3(self.fc1(out)))
        out = self.fc2(out)

        out = out.view(batch_size, -1)
        out = out[:,-1]

        return out, hidden

#Display network
#model = PpgNet()



################### GSR Net ###################
class GSRNet(nn.Module):

    def __init__(self, filters = 32, hidden_dim = 64, n_layers =2, drop = 0.25):
        super(GSRNet, self).__init__()
        
        self.ip_dim = filters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop = drop

        #Convolutional layers
        self.conv1 = nn.Conv1d(in_channels =  1, out_channels = filters, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv1d(in_channels = filters, out_channels = filters, kernel_size = 3, padding=1)

        #Max pooling layer (4)
        self.pool = nn.MaxPool1d(kernel_size = 4, stride = 4) 

        #Batch normalization
        self.batch1 = nn.BatchNorm1d(num_features = filters)
        self.batch2 = nn.BatchNorm1d(num_features = filters)
        self.batch3 = nn.BatchNorm1d(num_features = int(hidden_dim/4))

        #Dense & classification layer 
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim/4))
        self.fc2 = nn.Linear(int(hidden_dim/4), 1)


        #LSTM layer
        self.lstm = nn.LSTM(input_size = filters, 
                                hidden_size = hidden_dim, 
                                num_layers = n_layers, 
                                dropout = drop,
                                batch_first= True)
    
    def InitHiddenstate(self, batch_size):
           
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

    def forward(self, x, hidden): 

        batch_size = x.shape[0]
        #Convolutional block
        x = self.pool(F.relu(self.batch1(self.conv1(x)))) #First block
        x = self.pool(F.relu(self.batch2(self.conv2(x)))) #Second block
        
        #LSTM-block
        x = torch.transpose(x, 1, 2)
        out, hidden = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)

        #Prediction block
        out = F.relu(self.batch3(self.fc1(out)))
        out = self.fc2(out)

        out = out.view(batch_size, -1)
        out = out[:,-1]

        return out, hidden
#Display network
#GsrNet()



################### Multi-modular Net ###################



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

#Network()