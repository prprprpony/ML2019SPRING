import sys
import csv
import numpy as np
# preprocess data
fin = open(sys.argv[1],'r')
fout = open(sys.argv[2],'w')
raw_data = list(csv.reader(fin,delimiter=','))
K = 9 * 18 + 1 # 9 hours * 18 fatures + bias
w = np.load(sys.argv[3])
f = csv.writer(fout, delimiter=',')
f.writerow(['id','value'])
for i in range(0,len(raw_data),18):
    x = []
    for j in range(18):
        x += raw_data[i+j][6:11]
    s = raw_data[i][0]
    x += [1]
    for j in range(len(x)):
        if x[j] == 'NR':
            x[j] = '0'
    x = np.array(x,dtype='float64')
    y = np.dot(w,x)
    f.writerow([s, y])
