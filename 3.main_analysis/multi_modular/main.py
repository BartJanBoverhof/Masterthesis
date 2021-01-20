"""
Single-modular deep neural network
"""

################### 0. Prerequisites ###################

#Loading packages
import torch #PyTorch deep-learning library
from torch import nn #PyTorch additionals
from torch import optim #PyTorch optimizer for network training
import torch.nn.functional as F #PyTorch pre specified function library


x = torch.randn(10, 10)
print(x)
