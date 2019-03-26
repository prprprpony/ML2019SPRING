import numpy as np
import pandas as pd
from keras.models import load_model
import os, sys, csv
# preprocess data
x_data = pd.read_csv(os.popen("tail -n +2 " + sys.argv[1] + " | sed 's/,/ /g'"), sep=' ', header=None).values
x_data = np.delete(x_data,0,1) / 255
x_data = x_data.reshape((x_data.shape[0],48,48,1))

N = x_data.shape[0]
model = load_model(sys.argv[3])
y = model.predict_classes(x_data)

f = csv.writer(open(sys.argv[2],'w'), delimiter=',')
f.writerow(['id','label'])
for i in range(N):
    f.writerow([i, y[i]])
