import csv
import numpy as np
from sklearn.linear_model import LinearRegression
# preprocess data
raw_data = list(csv.reader(open('../train.csv',encoding='big5'),delimiter=','))
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
for i in range(18):
    for j in range(len(concat_data[i])):
        if concat_data[i][j] < 0:
            concat_data[i][j] = concat_data[i][j-1]
x_data = []
y_data = []
for base in range(0,len(concat_data[0]),20*24): # 20 days
    for i in range(20*24-9):
        x = []
        for j in range(18):
            x += concat_data[j][base+i:base+i+9]
        x_data.append(x+[1])
        y_data.append(concat_data[9][base+i+9])
x_data = np.array(x_data)
y_data = np.array(y_data)
N = 471 * 12
K = 9 * 18 + 1 # 9 hours * 18 fatures + bias
print(N,K)
assert(x_data.shape == (N, K))
assert(y_data.shape == (N, ))
# 
w = np.matmul(np.linalg.pinv(x_data),y_data)
cur_y = np.matmul(x_data, w)
delta_y = y_data - cur_y
loss = np.sqrt(1/N * np.sum(np.square(delta_y)))
print(w)
print('loss =',loss)
w = np.linalg.lstsq(x_data, y_data)[0]
cur_y = np.matmul(x_data, w)
delta_y = y_data - cur_y
loss = np.sqrt(1/N * np.sum(np.square(delta_y)))
print(w)
print('loss =',loss)
w = np.matmul(np.linalg.inv(np.matmul(x_data.transpose(),x_data)), np.matmul(x_data.transpose(), y_data))
cur_y = np.matmul(x_data, w)
delta_y = y_data - cur_y
loss = np.sqrt(1/N * np.sum(np.square(delta_y)))
print(w)
print('loss =',loss)
lin_reg = LinearRegression()
lin_reg.fit(x_data, y_data)
cur_y = lin_reg.predict(x_data)
delta_y = y_data - cur_y
loss = np.sqrt(1/N * np.sum(np.square(delta_y)))
print('loss =',loss)
#np.save('weight', w)
#np.save('weight', w)
