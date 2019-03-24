import sys
import csv
import numpy as np
# preprocess data
label = None
for x in csv.reader(open('../X_test')):
    label = x
    break
weight = np.load('weight.npz')
w = weight['w'][1:]
mean = weight['mean']
std = weight['std']
K = len(label)
a = []
for i in range(K):
    print(label[i], '\t\t\t\t', w[i])
    a.append((w[i],label[i]))
a.sort()
print('smallest')
for i in range(10):
    print(a[i])
print('largest')
for i in range(10):
    print(a[-i-1])
