#!/usr/bin/env python3
import sys,os
import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 

model = resnet50(pretrained=True).cuda()
model.eval()
criterion = nn.CrossEntropyLoss()
t_mean = np.array([0.485, 0.456, 0.406])
t_std = np.array([0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=t_mean, std=t_std)])
r_mean = -t_mean / t_std
r_std = 1 / t_std
rev = transforms.Compose([transforms.Normalize(mean=r_mean, std=r_std),
                                   lambda x: torch.clamp(x, 0., 1.),
                                   transforms.ToPILImage()])

input_dir = sys.argv[1]
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

N = 200
bs = 50
for start in range(0,N,bs):
    image = []
    for i in range(start,min(N,start+bs)):
        im = Image.open(os.path.join(input_dir, '%03d.png' % i)).convert('RGB')
        im = trans(im)
        image.append(im.unsqueeze(0))
    image = torch.cat(image).cuda()


    epsilon = 1e-4
    alpha = 5e-3
    epochs = 17

    target = torch.zeros(image.shape[0],dtype=torch.long).cuda()
    image.requires_grad_()
    eta = torch.zeros(image.shape).cuda()

    for _ in range(epochs):
        print(start//bs,_)
        output = model(image + eta)
        loss = criterion(output, target)
        loss.backward()

        eta += -alpha * torch.sign(image.grad.data)
        image.grad.data.zero_()

    adv = (image + eta).cpu().detach()
    print(adv.shape)
    for i in range(start,min(N,start+bs)):
        im = adv[i - start]
        im = im.squeeze()
        im = rev(im)
        im.save(os.path.join(output_dir, '%03d.png' % i))


