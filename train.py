# Imports here
import sys
import json
import numpy as np
import argparse
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Load arguments
parser = argparse.ArgumentParser(description = "Parser of train code")

parser.add_argument('data_dir', help = 'Provide data directory.', type = str)
parser.add_argument ('--save_dir', help = 'Provide save directory (default = directory of script).', type = str, default = 'curr_dir')
parser.add_argument ('--arch', help = 'Provide the type of model (alexnet or vgg16) (default = VGG16).', type = str, default = 'vgg16')
parser.add_argument ('--lr', help = 'Provide the learning rate (default = 0.001).', type = float, default = 0.001)
parser.add_argument ('--hidden_units', help = 'Provide the Classifier hidden untis (default = 1024).', type = int, default = 1024)
parser.add_argument ('--epochs', help = 'Provide the number of epochs (default = 3).', type = int, default = 3)
parser.add_argument ('--gpu', help = "Provide option to use gpu (default = false/cpu).", action='store_true')

args = parser.parse_args()

data_dir = args.data_dir
arch = args.arch
lr = args.lr
hidden_units = args.hidden_units
epochs = args.epochs

if args.gpu:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# Load data directories
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#print(cat_to_name) # Test cat_to_name

# TODO: Build and train your network
if arch == 'alexnet':
    model = models.alexnet(pretrained=True)
    
    input_units = 9216
        
elif arch == 'vgg16':
    model = models.vgg16(pretrained=True)


    input_units = 25088
    
else:
    sys.exit('Arch selected is not supported.')
    
#print(model) # Check model
#sys.exit('Stop to check model.')

# Freeze parameters so we don't backprop through them
for param in model.parameters():
        param.requires_grad = False

# Update the classifier
classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(input_units, 4096, bias=True)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p = 0.5)),
                            ('fc2', nn.Linear(4096, hidden_units, bias=True)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout (p = 0.5)),
                            ('fc3', nn.Linear (hidden_units, 102, bias=True)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    
model.classifier = classifier

# Set criterion, optimizer and GPU
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)

model.to(device)
#print(model) # Check new model
#sys.exit('Stop to check new model.')

# Rubric Training the Network (1)
steps = 0
print_every = 40

for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in train_loader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        train_logps = model.forward(inputs)
        loss = criterion(train_logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            valid_accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    valid_logps = model.forward(inputs)
                    batch_loss = criterion(valid_logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(valid_logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Rubric Validation Loss and Accuracy (2)
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                  f"Validation accuracy: {valid_accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()
            
# Rubric Saving the Model (3)
# TODO: Save the checkpoint
model.class_to_idx = train_data.class_to_idx

checkpoint = {'classifier': model.classifier,
              'arch': arch,
              'mapping': model.class_to_idx,
              'state_dict': model.state_dict()}

if args.save_dir == 'curr_dir':
    torch.save(checkpoint, 'checkpoint.pth')
else:
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
