import csv
import pandas as pd
import numpy as np
# preprocess data
raw_data = list(csv.reader(open('train.csv',encoding='big5'),delimiter=','))
raw_data = raw_data[1:] 
for i in range(len(raw_data)):
    raw_data[i] = raw_data[i][3:]
    for j in range(len(raw_data[i])):
        if raw_data[i][j] == 'NR':
            raw_data[i][j] = '0'
        raw_data[i][j] = float(raw_data[i][j])
concat_data = [[] for i in range(18)]
for i in range(len(raw_data)):
    concat_data[i % 18] += raw_data[i]
x_data = []
y_data = []
for base in range(0,len(concat_data[0]),20*24): # 20 days
    for i in range(20*24-9):
        x_data.append([])
        for j in range(18):
            x_data[-1] += concat_data[j][base+i:base+i+9]
        y_data.append(concat_data[9][base+i+9])
for i in range(len(x_data)):
    x_data[i].append(1)
x_data = np.array(x_data)
y_data = np.array(y_data)
N = 471 * 12
K = 9 * 18 + 1 # 9 hours * 18 fatures + bias
assert(x_data.shape == (N, K))
assert(y_data.shape == (N, ))
# 
w = np.ones(K) / K
lr = 1
T = 10**5
prev_gra = 0
xt = x_data.transpose()
for t in range(T):
    cur_y = np.matmul(x_data, w)
    delta_y = y_data - cur_y
    grad = -2/N * np.matmul(xt, delta_y)
    prev_gra += np.sum(np.square(grad))
    ada = np.sqrt(prev_gra)
    w -= lr / ada * grad
    if t % 1000 == 0:
        loss = 1/N * np.sum(np.square(delta_y))
        print(t,loss)
np.save('weight', w)
