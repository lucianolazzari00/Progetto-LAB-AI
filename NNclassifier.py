import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as data
from tqdm import tqdm

try:
    exec_mode = sys.argv[1]
except IndexError:
    print("Error! missing one argument \n -usage: NNclassifier.py [TRAIN|LOAD]")
    exit(0)

class ClassifierEasy(nn.Module):
    def __init__(self,n_in,n_hid,n_out):
        super().__init__()
        self.linear1 = nn.Linear(n_in,n_hid)
        self.act_f = nn.Tanh()
        self.linear2 = nn.Linear(n_hid,n_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_f(x)
        x = self.linear2(x)
        return x



class XORDataset(data.Dataset):

    def __init__(self, size, std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label




### training phase ###

def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    # Set model to train mode
    model.train() 
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:
            
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            # data_inputs = data_inputs.to(device)
            # data_labels = data_labels.to(device)
            
            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            
            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())
            
            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero. 
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad() 
            # Perform backpropagation
            loss.backward()
            
            ## Step 5: Update the parameters
            optimizer.step()


def eval_model(model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.
    
    with torch.no_grad(): # Deactivate gradients for the following code
        for data_inputs, data_labels in data_loader:
            
            # Determine prediction of model on dev set
            #data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1
            
            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]
            
    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

###--- initializing ds ---###

train_dataset = XORDataset(size=3000)
print("Size of dataset:", len(train_dataset))

###--- initializing data loader ---###

train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

if exec_mode == "TRAIN":
    ###--- initializing model ---###

    model = ClassifierEasy(2,4,1)
    print(model)

    ###--- TRAINING ---###

    loss_module = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    train_model(model, optimizer, train_data_loader, loss_module)

     ###--- SAVING MODEL ---###
    state_dict = model.state_dict()
    print(state_dict)

    torch.save(state_dict, "xor_model.tar")

if exec_mode == "LOAD":
    # Load state dict from the disk (make sure it is the same name as above)
    state_dict = torch.load("xor_model.tar")

    # Create a new model and load the state
    model = ClassifierEasy(2, 4, 1)
    model.load_state_dict(state_dict)

###---TESTING---###

test_dataset = XORDataset(size=500)
# drop_last -> Don't drop the last batch although it is smaller than 128
test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False) 

eval_model(model, test_data_loader)





###Utils
#for per i parametri
# for name,param in model.named_parameters():
#     #prova
#     print(name)
#     print("--")
#     print(param.shape)

# iterate over the ds trough the data loader
# i=0
# for batch in data_loader:
#     print("batch n: ", i)
#     i+=1
#     data_inputs, data_labels = next(iter(data_loader))
#     print("Data inputs", data_inputs.shape, "\n", data_inputs)
#     print("Data labels", data_labels.shape, "\n", data_labels)
