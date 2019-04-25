#!/usr/bin/env python3
import matplotlib.pyplot as plt
import sys,os
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(3)
np.random.seed(0)


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
adv_dir = sys.argv[2]
categories = pd.read_csv('../categories.csv')['CategoryName'].values

def f(data):
    image = []
    for x in data:
        image.append(trans(x).unsqueeze(0))
    image = torch.cat(image).cuda()
    sm = nn.Softmax(dim=1)
    output = sm(model(image))
    output = output.cpu().detach().numpy()
    idx = np.argsort(output)[:,-3:]
    ret = []
    for i in range(3):
        labels = categories[idx[i]].tolist()
        for j in range(3):
            labels[j] = labels[j].split(',')[0]
        ret.append([labels, output[i][idx[i]]])
    return ret


before = [Image.open(os.path.join(input_dir, '%03d.png' % i)).convert('RGB') for i in range(98,101)]
after = [Image.open(os.path.join(adv_dir, '%03d.png' % i)).convert('RGB') for i in range(98,101)]

before = f(before)
after = f(after)
y_pos = np.arange(3)
for i in range(3):
    y = before[i][1]
    obj = tuple(before[i][0])
    plt.bar(y_pos, y,align='center',alpha=0.5)
    plt.xticks(y_pos, obj)
    plt.ylabel('probability')
    plt.title('%03d.png before' % (98 + i))
    plt.savefig(f'before{i}.png')
    plt.clf()
for i in range(3):
    y = after[i][1]
    obj = tuple(after[i][0])
    plt.bar(y_pos, y,align='center',alpha=0.5)
    plt.xticks(y_pos, obj)
    plt.ylabel('probability')
    plt.title('%03d.png after' % (98 + i))
    plt.savefig(f'after{i}.png')
    plt.clf()
