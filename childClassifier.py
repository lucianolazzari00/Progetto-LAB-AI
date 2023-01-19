import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision import datasets, transforms

import numpy as np

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
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

#--- validationset

validationset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=False, transform=transform)
validation_loader = DataLoader(validationset, batch_size=64, shuffle=True)

##--- testset
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=False, train=False, transform=transform)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)

class MLP(nn.Module):
    def __init__(self,n_in=784 ,n_out=2, hidden = [256,128,64]):
        super().__init__()
        self.linear1 = nn.Linear(n_in,hidden[0])
        self.l1_drop = nn.Dropout(0.2)
        self.linear2 = nn.Linear(hidden[0],hidden[1])
        self.l2_drop = nn.Dropout(0.2)
        self.linear3 = nn.Linear(hidden[1],hidden[2])
        self.l3_drop = nn.Dropout(0.2)
        self.linear4 = nn.Linear(hidden[2], n_out)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.l1_drop(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.l2_drop(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.l3_drop(x)
        return F.softmax(self.linear4(x), dim=1)


def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.view(data.shape[0], -1)
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))


def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.view(data.shape[0], -1)
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))


model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()

epochs = 5
lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)

# -------- single prediction
# dataiter = iter(test_loader)
# _data = next(dataiter)
# print("-Expexted prevision: ")
# print(_data[1][0])
# single_data = _data[0][0].view(_data[0][0].shape[0], -1)

# def predict(input):
#     return model(input).max(1)[1]

# print("-Model prevision:", end="")
# print(predict(single_data))