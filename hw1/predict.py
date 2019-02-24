import csv
import numpy as np
# preprocess data
raw_data = list(csv.reader(open('test.csv'),delimiter=','))
K = 9 * 18 + 1 # 9 hours * 18 fatures + bias
w = np.load('weight.npy')
f = csv.writer(open('ans.csv','w'), delimiter=',')
f.writerow(['id','value'])
print(w)
for i in range(0,len(raw_data),18):
    x = []
    for j in range(18):
        x += raw_data[i+j][2:]
    x += [1]
    for j in range(len(x)):
        if x[j] == 'NR':
            x[j] = '0'
    x = np.array(x,dtype='float64')
    y = np.dot(w,x)
    f.writerow(['id_'+str(i//18), y])
