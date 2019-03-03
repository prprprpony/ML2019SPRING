import sys
import csv
import pickle
import numpy as np
# preprocess data
fin = open(sys.argv[1],'r')
fout = open(sys.argv[2],'w')
raw_data = list(csv.reader(fin,delimiter=','))
K = 9 * 18 + 1 # 9 hours * 18 fatures + bias
svm_reg = pickle.load(open(sys.argv[3],'rb'))
f = csv.writer(fout, delimiter=',')
f.writerow(['id','value'])
x_data = []
s_arr = []
for i in range(0,len(raw_data),18):
    x = []
    for j in range(18):
        x += raw_data[i+j][2:]
    s = raw_data[i][0]
    for j in range(len(x)):
        if x[j] == 'NR':
            x[j] = '0'
    x.append(1)
    x_data.append(x)
    s_arr.append(s)
x_data = np.array(x_data,dtype='float64')
y_data = svm_reg.predict(x_data)
assert(len(y_data) == len(s_arr))
for i in range(len(y_data)):
    f.writerow([s_arr[i], y_data[i]])
