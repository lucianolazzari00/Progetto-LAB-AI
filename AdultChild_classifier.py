import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import time
import os
import sys
from torch.optim import lr_scheduler
import random
import time
from PIL import Image

try:
    exec_mode = sys.argv[1]
except IndexError:
    print("Error! missing one argument \n -usage: NNclassifier.py [TRAIN|LOAD]")
    exit(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on device: {device}')
##=======dataset=========

####### TODO: try different normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}

data_dir = 'face_dataset/UTKFace'

train_dataset = ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
val_dataset = ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
test_dataset = ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])


batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(f'\n---------------------------train_dataset\n: {train_dataset}')
print(f'\n---------------------------val_dataset\n: {val_dataset}')
print(f'\n---------------------------test_dataset\n: {test_dataset}')
print('----------------------------------------')
##=======================

# # visualize a batch
# for images,labels in train_dataloader:
#     print(labels)
#     for i in range(len(images)):
#         # Convert the image tensor to a PIL image
#         image = images[i]
#         image = transforms.ToPILImage()(image)
#         image.show()
#     break

for images,labels in train_dataloader:
    print(images.shape)
    break

##=======MODEL=========

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        #print(x.shape)
        x = x.view(-1, 256 * 12 * 12)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


##=========TRAINING=========

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    # Move the model to the GPU if available
    model = model.to(device)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        i=0
        for inputs, labels in train_loader:
            if i%500==0:
                print(f'------------\ntraining: ..... {i}/18695')
                if i!=0:
                    print(f'curr loss: {running_loss/i*batch_size}')
            i+=1
            if i > 500:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        
        print("Epoch: {}/{}.. ".format(epoch+1, num_epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Validation Loss: {:.3f}.. ".format(val_loss),
              "Validation Accuracy: {:.3f}".format(val_acc))
    return model, train_losses, val_losses

##========TESTING=======

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(predicted)):
                if(predicted[i]!=labels[i]):
                    print(f'correct: {labels[i]} | predicted: {predicted[i]}')
    test_loss /= len(test_loader)
    test_acc = correct / total
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
    return test_loss, test_acc

#start a little "demo":
# take a batch, plot images and make predictions
def demo(model):
    for images,labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        #print(f'[-]shape: {images.shape}')
        #print(labels)
        for i in range(len(images)):
            image = images[i]
            # Forward pass
            output = model(image)
            # Get the class with highest probability
            _, pred = torch.max(output, 1)
            print(f'[PREDICT]: {pred.cpu().numpy()[0]}')
            print(f'[CORRECT]: {labels[i]}')

            image = transforms.ToPILImage()(image)
            image.show()
        break

##======PREDICTIONS=====

data_transf = transforms.Compose([
        transforms.Resize((200,200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def predict(model, image, data_transform=data_transf):
    # Preprocess the image
    image = Image.fromarray(image)
    image = data_transform(image)
    
    imagesh = transforms.ToPILImage()(image)
    imagesh.show()
    # Convert to a batch of size 1
    #image = image.unsqueeze(0)
    
    # Move the input tensor to the appropriate device
    image = image.to(device)
    
    # Forward pass
    output = model(image)
    print(output)
    # Get the class with highest probability
    _, pred = torch.max(output, 1)
    
    return pred.item()

##=========MAIN=========

def main():
    # Initialize the model
    model = ComplexCNN()
    # Initialize the criterion (loss function)
    criterion = nn.CrossEntropyLoss()
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if exec_mode == "TRAIN":
        # Train the model
        trained_model, train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=4)
        # save the model
        state_dict = trained_model.state_dict()
        torch.save(state_dict, "adch_1_model.tar")
        # Test the model
        test_model(trained_model, test_dataloader, criterion)
    
    if exec_mode == "LOAD":
        # Load state dict from the disk (make sure it is the same name as above)
        state_dict = torch.load("adch_model.tar", map_location=torch.device('cpu')) ####LEVA LA CPUU!! TODO
        #state_dict = torch.load("adch_model.tar")

        # Create a new model and load the state
        trained_model = ComplexCNN()
        trained_model.load_state_dict(state_dict)
        trained_model = trained_model.to(device)
        #test_model(trained_model, test_dataloader, criterion)
        
        demo(trained_model)
    
if __name__ == '__main__':
    main()
