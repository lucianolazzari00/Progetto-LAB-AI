import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Subset
import time
import os
import copy
from torch.optim import lr_scheduler
import random
import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

######################################################

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
######################################################

##learning with : 45000train, 9000 validate


##--------dataset------
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# Select the indices of the samples you want to include in the subset
indices_train = random.sample(range(len(trainset)), 6000)

# Create the subset dataset
sub_trainset = Subset(trainset, indices_train)

trainloader = torch.utils.data.DataLoader(sub_trainset, batch_size=4,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

indices_test = random.sample(range(len(testset)), 1000)

# Create the subset dataset
sub_testset = Subset(testset, indices_test)

testloader = torch.utils.data.DataLoader(sub_testset, batch_size=4,
                                         shuffle=False)

dataloaders = {'train': trainloader, 'val' : testloader}
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataset_sizes = {'train':6000 , 'val':1000}

print("current device: ",end="")
print(device)
##---------------------

## -------train-------
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            d = 0
            for inputs, labels in dataloaders[phase]:
                if False:
                    print(f'[X] current data idx: {d}')
                d+=1

                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

## -------------------
model_conv = torchvision.models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 10)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


##single prediction
#for inputs,labels in dataloaders['val']:
#    inputs = inputs.to(device)
#    labels = labels.to(device)
#    # print images
#    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

#    outputs = model_conv(inputs)
#    print(outputs)
#    _, predicted = torch.max(outputs, 1)
#    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                                for j in range(4)))
#    break

train = False

if train:
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=30)
    ###--- SAVING MODEL ---###
    state_dict = model_conv.state_dict()
    print(state_dict)

    torch.save(state_dict, "cifar10_test_model.tar")
else:
    # Load state dict from the disk (make sure it is the same name as above)
    state_dict = torch.load("cifar10_model.tar")

    # Create a new model and load the state
    model_conv = torchvision.models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 10)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    model_conv.load_state_dict(state_dict)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=1)


i = 0
##single prediction
for inputs,labels in dataloaders['val']:
    inputs = inputs.to(device)
    labels = labels.to(device)
    # print images
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = model_conv(inputs)
    #print(outputs)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))
    i+=1
    if i%5 == 0: 
        break