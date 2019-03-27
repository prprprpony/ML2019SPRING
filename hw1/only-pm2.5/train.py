import csv
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
        x_data.append(concat_data[9][base+i:base+i+9])
        y_data.append(concat_data[9][base+i+9])
for i in range(len(x_data)):
    x_data[i].append(1)
x_data = np.array(x_data)
y_data = np.array(y_data)
N = 471 * 12
K = 9 + 1 # 9 hours + bias
assert(x_data.shape == (N, K))
assert(y_data.shape == (N, ))
# 
w = np.matmul(np.linalg.pinv(x_data),y_data)
cur_y = np.matmul(x_data, w)
delta_y = y_data - cur_y
loss = np.sqrt(1/N * np.sum(np.square(delta_y)))
print('loss =',loss)
np.save('only-pm2.5-weight', w)
