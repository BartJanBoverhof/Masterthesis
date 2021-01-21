"""
Single-modular deep neural network EEG modality.
"""



################### 0. Prerequisites ###################
#Loading packages
import torch #PyTorch deep-learning library
from torch import nn, optim #PyTorch additionals and training optimizer
import torch.nn.functional as F #PyTorch library providing a lot of pre-specified functions



################### 1. Loading data ###################
#Load the training data
train_dat = 
train_loader = torch.utils.data.DataLoader(train_dat, batch_size = 64, shuffle = True)

#Load the test data
test_dat = 
test_loader = torch.utils.data.Dataloader(test_dat, batch_size = 64, shuffle = True)



################### 2. Define Network architecture ###################
class Net(nn.Module):
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
Net()


