"""
@Author: Bart-Jan Boverhof
@Last Modified by: Bart-Jan Boverhof
@Description Loading the data and training all networks.
"""



################### 0. Prerequisites ###################
#Loading packages
import torch #PyTorch deep-learning library
from torch import nn, optim #PyTorch additionals and training optimizer
import torch.nn.functional as F #PyTorch library providing a lot of pre-specified functions



################### 1. Loading data ###################
#Training data
train_dat_eeg = 
train_dat_ppg =
train_dat_gsr = 

train_loader = torch.utils.data.DataLoader(train_dat, batch_size = 64, shuffle = True)