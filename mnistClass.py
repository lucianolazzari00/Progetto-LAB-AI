import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision import datasets, transforms

import numpy as np
# import sklearn
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt

# This is when we use GPU instance to run colab. 
# How to change Runtime in colab?? 
# Go to Runtime-> Change Runtime type -> Hardware Accelerator -> GPU

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

### DATASETS INITIALIZATION ###

#--- trainset

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('../data', download=True, train=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

#--- validationset

validationset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=False, transform=transform)
validation_loader = DataLoader(validationset, batch_size=64, shuffle=True)

##--- testset
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=False, transform=transform)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)

print(trainset)
print(validationset)
print(testset)