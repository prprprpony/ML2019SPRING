import sys
import csv
import numpy as np
from sklearn.externals.joblib import dump, load
# preprocess data
x_data = np.genfromtxt(sys.argv[1], delimiter=',')[1:,]
N = x_data.shape[0]
model = load(sys.argv[3])
y = model.predict(x_data)

f = csv.writer(open(sys.argv[2],'w'), delimiter=',')
f.writerow(['id','label'])
for i in range(N):
    f.writerow([i+1, y[i]])
