#!/usr/bin/env python3
import sys,os
import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 

model = resnet50(pretrained=True)
model.eval()
criterion = nn.CrossEntropyLoss()
input_dir = sys.argv[1]
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
epsilon = 1e-2
epochs = 170
for filename in os.listdir(input_dir):
    if filename[-4:] != '.png':
        continue
    image = Image.open(os.path.join(input_dir, filename)).convert('RGB')
    trans = transforms.Compose([transforms.ToTensor()])
    rev = transforms.ToPILImage()
    image = trans(image)
    image = image.unsqueeze(0)
    image.requires_grad = True
    zero_gradients(image)
    output = model(image)
    loss = criterion(output, torch.tensor([0]))
    loss.backward()

    adv = image - epsilon * image.grad.sign_()
    adv = adv.squeeze()
    adv = rev(adv)
    adv.save(os.path.join(output_dir, filename))


