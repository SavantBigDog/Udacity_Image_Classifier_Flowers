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
from PIL import Image

# Load arguments
parser = argparse.ArgumentParser(description = "Parser of predict code.")

parser.add_argument ('--check_file', help = 'Provide file for checkpoint (default = directory of script).', type = str, default = 'checkpoint.pth')
parser.add_argument ('--image_dir', help = 'Provide path to image (default = directory of script).', type = str)
parser.add_argument ('--category_names', help = 'Provide a json file with categories (default = cat_to_name.json).', type = str, default = 'cat_to_name.json')
parser.add_argument ('--top_k', help = 'Provide topk number of classes (default = 1).', type = int, default = 1)
parser.add_argument ('--gpu', help = "Provide option to use gpu (default = false/cpu).", action='store_true')

args = parser.parse_args()

if args.image_dir:
    image_path = args.image_dir
else:
    image_path = 'flowers/test/100/image_07896.jpg'
    
if args.category_names:
    cat_names = args.category_names
else:
    cat_names = 'cat_to_name.json'

topk = args.top_k

if args.gpu:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# Load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained=True)
    else:
        model = models.vgg16 (pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['mapping']
    model.load_state_dict = checkpoint['state_dict']
    
    return model

model = load_checkpoint(args.check_file)

#print(model) # Check model
#sys.exit('Stop to check model.')

with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

# Process image function
def process_image(image):
    img = Image.open(image)
    
    # Citation Udacity Mentor Survesh and Arun answers in Ask a Mentor
    width = 256
    height = 256
    img = img.resize((width, height))
    
    crop_width = 224
    crop_height = 224
    
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = (width + crop_width) / 2
    bottom = (height + crop_height) / 2
    img = img.crop((left, top, right, bottom))
    
    np_img = np.array(img)/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_img = (np_img - mean) / std
    
    # TODO: Process a PIL image for use in a PyTorch model
    np_img = np_img.transpose((2, 0, 1))
       
    return torch.from_numpy(np_img)

# Predict image function
def predict(image, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Citation Udacity Student Renata H and Mentor Ratul answer in Ask a Mentor.
    # Renata was stuck in the same spot that I was at about the same time.
    model.to(device)
    model.eval()
    names = []
    
    image_to_predict = process_image(image)
    image_to_predict = image_to_predict.unsqueeze(0)
    image_to_predict = image_to_predict.float()
    
    with torch.no_grad():
        output = model.forward(image_to_predict.to(device))
    
    probability = F.softmax(output.data,dim=1)
    top_probability, indices = torch.topk(probability, dim=1, k=topk)
    indices = np.array(indices)
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [index_to_class[each] for each in indices[0]]
    
    for classes in top_classes:
            names.append(cat_to_name[str(classes)])
  
    return top_probability.cpu().numpy(), top_classes

top_probability, top_classes = predict(image_path, model, device, topk)

class_names = [cat_to_name [item] for item in top_classes]

new_probability = top_probability.flatten()

for i in range (topk):
     print("Number: {}/{}  ".format(i+1, topk),
           "Flower name: {}  ".format(class_names [i]),
           "Probability: {:.3%}  ".format(new_probability [i])
           )
#print(new_probability) #Check and compare results
#print(top_classes) #Check and compare results