#!/usr/bin/env python3
# standard library
import argparse
import csv
import time
import sys
import os
# other library
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# PyTorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data 

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w


class Dataset(data.Dataset):
    def __init__(self, imgae_dir):
        self.total_img = []
        for i in range(1, 40001):
        #for i in range(1, 40):
            print("loading image %d/40000" % i, end='\r')
            fname = os.path.join(imgae_dir, "%06d.jpg" % (i))
            img = Image.open(fname)
            img.load()
            row = np.asarray(img)
            self.total_img.append(row)

        # since at pytorch conv layer, input=(N, C, H, W)
        self.total_img = np.transpose(np.array(self.total_img, dtype=float), (0, 3, 1, 2))
        # normalize
        self.total_img = (self.total_img ) / 255.0
        print("=== total image shape:",  self.total_img.shape)
        # shape = (40000, 3, 32, 32)

    def __len__(self):
        return len(self.total_img)

    def __getitem__(self, index):
        return(self.total_img[index])

class Net(nn.Module):
    def __init__(self, image_shape, latent_dim):
        super(Net, self).__init__()
        self.shape = image_shape
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3,padding=1), # 16, 11 11
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16, 5, 5
            # TODO: define your own structure
            nn.Conv2d(16, 8, 3, stride=2, padding=1), # 8, 3, 3
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1) # 8, 2, 2
        )
        # assume output shape is (Batch, C, H, W)
        # N = C * H * W
        self.N = 8 * 2 * 2

        self.fc1 = nn.Linear(self.N, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.N)

        self.decoder = nn.Sequential(
            # TODO: define yout own structure
            nn.ConvTranspose2d(8, 16, 3, stride=2),  
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 6, stride=2, padding=1), 
        )

    def forward(self, x):
        x = self.encoder(x)
        # flatten
        x = x.view(len(x), -1)
        encoded = self.fc1(x)

        x = F.relu(self.fc2(encoded))
        x = x.view(-1, 8,2,2)
        x = self.decoder(x)
        return encoded, x

def training(train, val, model, device, n_epoch, batch, save_name, lr):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('=== start training, parameter total:%d, trainable:%d' % (total, trainable))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                             weight_decay=1e-5)

    for epoch in range(n_epoch):
        total_loss, best_loss = 0, 100

	# training set
        model.train()
        for idx, image in enumerate(train):
            image = image.to(device, dtype=torch.float)
            _, reconsturct = model(image)
            loss = criterion(reconsturct, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += (loss.item() / len(train))

            print('[Epoch %d | %d/%d] loss: %.4f' %
                 ((epoch+1), idx*batch, len(train)*batch, loss.item()), end='\r')
        print("\n  Training  | Loss:%.4f " % total_loss)

	# validation set
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for idx, image in enumerate(val):
                    image = image.to(device, dtype=torch.float)
                    _, reconstruct = model(image)

                    loss = criterion(reconstruct, image)
                    total_loss += (loss.item() / len(val))

            print(" Validation | Loss:%.4f " % total_loss)
        # save model
        if total_loss < best_loss:
                best_loss = total_loss
                print("saving model with loss %.4f...\n" % total_loss)
                torch.save(model.state_dict(), "%s" % save_name)

def clustering(model, device, loader, n_iter, reduced_dim):
    model.eval()
    latent_vec = torch.tensor([]).to(device, dtype=torch.float)
    for idx, image in enumerate(loader):
        print("predict %d / %d" % (idx+1, len(loader)) , end='\r')
        image = image.to(device, dtype=torch.float)
        latent, r = model(image)
        latent_vec = torch.cat((latent_vec, latent), dim=0)

    latent_vec = latent_vec.cpu().detach().numpy()
    print('\n',latent_vec.shape)

    # shape = (40000, latent_dim)

    # tsne = TSNE(n_components=reduced_dim, verbose=1, perplexity=50, n_iter=n_iter)
    # latent_vec = tsne.fit_transform(latent_vec)

    # pca = PCA(n_components=reduced_dim, copy=False, whiten=True, svd_solver='full')
    # latent_vec = pca.fit_transform(latent_vec)

    kmeans = KMeans(n_clusters=2, random_state=20190523, max_iter=n_iter).fit(latent_vec)
    return kmeans.labels_


def read_test_case(path):
    dm = pd.read_csv(path)
    img1 = dm['image1_name']
    img2 = dm['image2_name']
    test_case = np.transpose(np.array([img1, img2]))
    return test_case

def prediction(label, test_case, output):
    result = []
    for i in range(len(test_case)):
        index1, index2 = int(test_case[i][0])-1, int(test_case[i][1])-1
        if label[index1] != label[index2]:
            result.append(0)
        else:
            result.append(1)

    result = np.array(result)
    with open(output, 'w') as f:
        f.write("id,label\n")
        for i in range(len(test_case)):
            f.write("%d,%d\n" % (i, result[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--image_dir', default='', type=str)
    parser.add_argument('--test_case', default='', type=str)
    parser.add_argument('--output_name', default='', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--load_state', default='', type=str)
    args = parser.parse_args()

    all_data = Dataset(args.image_dir)
    Nval = int(len(all_data) * 0.2)
    val = all_data[:Nval]
    train = all_data[Nval:]
   
    model = Net((3,32,32), args.latent_dim)
    model.cuda()
    device = torch.device('cuda')
    loader = data.DataLoader(all_data, batch_size=args.batch,num_workers=10)
    train = data.DataLoader(train, batch_size=args.batch,num_workers=10)
    val = data.DataLoader(val, batch_size=args.batch,num_workers=10)
    if len(args.load_state):
        model.load_state_dict(torch.load(args.load_state))

    if len(args.output_name):
        test_case = read_test_case(args.test_case)
        label = clustering(model, device, loader, 300, args.latent_dim)
        prediction(label, test_case, args.output_name)
    else:
        training(train, val, model, device, args.epoch, args.batch, args.model_name, args.lr)
    


