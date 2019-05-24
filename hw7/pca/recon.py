#!/usr/bin/env python3
import os
import sys
import numpy as np 
from skimage.io import imread, imsave
IMAGE_PATH = sys.argv[1]

# Images for compression & reconstruction
test_image = [sys.argv[2]]

# Number of principal components used
k = 5
def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M
filelist = [f'{i}.jpg' for i in range(415)]

# Record the shape of images
img_shape = (600, 600, 3)

img_data = []
for filename in filelist:
    tmp = imread(os.path.join(IMAGE_PATH,filename))  
    img_data.append(tmp.flatten())

training_data = np.array(img_data).astype('float32')

# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 

# Use SVD to find the eigenvectors 
eigface, eigval, v = np.linalg.svd(training_data.T, full_matrices = False)  


eigfk = eigface[:,:k]

'''1c'''
for x in test_image: 
    # Load image & Normalize
    picked_img = imread(os.path.join(IMAGE_PATH,x))  
    X = picked_img.flatten().astype('float32') 
    X -= mean

    # Compression
    weight = np.dot(X, eigfk)

    # Reconstruction
    reconstruct = np.zeros(600*600*3)
    for i in range(k):
        reconstruct += eigfk[:,i] * weight[i]
    reconstruct = process(reconstruct + mean)
    imsave(sys.argv[3], reconstruct.reshape(img_shape)) 


