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
epsilon = 1e-4
alpha = 1e-4
epochs = 170
cnt = 0


for filename in os.listdir(input_dir):
    if filename[-4:] != '.png':
        continue
    cnt += 1
    image = Image.open(os.path.join(input_dir, filename)).convert('RGB')
    target = torch.tensor([0])
    trans = transforms.Compose([transforms.ToTensor()])
    rev = transforms.ToPILImage()
    image = trans(image)
    image = image.unsqueeze(0)
    image.requires_grad_()
    eta = torch.zeros(image.shape)

    for _ in range(epochs):
        print(cnt,_)
        output = model(image + eta)
        loss = criterion(output, target)
        loss.backward()

        eta += -alpha * torch.sign(image.grad.data)
        image.grad.data.zero_()

    adv = image + eta
    adv = adv.squeeze()
    adv = rev(adv)
    adv.save(os.path.join(output_dir, filename))


