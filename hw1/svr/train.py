import csv
import pickle
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
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
x_data = []
y_data = []
for base in range(0,len(concat_data[0]),20*24): # 20 days
    for i in range(20*24-9):
        x = []
        for j in range(18):
            x += concat_data[j][base+i:base+i+9]
        flag = False
        for v in x:
            if v < 0:
                flag = True
        if flag:
            continue
        x_data.append(x+[1])
        y_data.append(concat_data[9][base+i+9])
x_data = np.array(x_data)
y_data = np.array(y_data)
#N = 471 * 12
N = len(x_data)
K = 9 * 18 + 1 # 9 hours * 18 fatures + bias
print(N,K)
assert(x_data.shape == (N, K))
assert(y_data.shape == (N, ))
# 
#svm_reg = LinearSVR(C=0.001)
svm_reg = LinearSVR(C=0.001)
'''
param_grid = dict(kernel=['rbf', 'linear','poly','sigmoid'], C=range(1,100), gamma=np.arange(1e-4,1e-2,0.0001).tolist())
svm_reg = SVR()
grid = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(x_data,y_data)
'''
svm_reg.fit(x_data,y_data)
cur_y = svm_reg.predict(x_data)
delta_y = y_data - cur_y
loss = np.sqrt(1/N * np.sum(np.square(delta_y)))
print('loss =',loss)
pickle.dump(svm_reg,open('svr-'+str(loss)+'.pickle','wb'))
