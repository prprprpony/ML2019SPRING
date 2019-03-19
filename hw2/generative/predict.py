import sys
import csv
import numpy as np
# preprocess data
x_data = np.genfromtxt(sys.argv[1], delimiter=',')[1:,]
N = x_data.shape[0]
weight = np.load(sys.argv[3])
w = weight['w']
mean = weight['mean']
std = weight['std']
b = weight['b']
x_data = (x_data - mean) / std
z = np.matmul(x_data, w) + b
fw = 1/(1+np.exp(-z))
y = (fw>0.5).astype(int)

f = csv.writer(open(sys.argv[2],'w'), delimiter=',')
f.writerow(['id','label'])
for i in range(N):
    f.writerow([i+1, y[i]])
