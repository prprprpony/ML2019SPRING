#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os, sys, csv
# preprocess data
MAXV = 5000
MAX_LENGTH=30
def pre(fn):
    token = pickle.load(open('token.pickle','rb'))
    ret = [line for line in open(fn,'r')]
    ret = token.texts_to_sequences(ret)
    for i in range(len(ret)):
        a = np.zeros(MAXV+2)
        for j in ret[i]:
            if j > 0:
                a[j] += 1
        ret[i] = a
    return np.array(ret)

x_data = pre(sys.argv[3])
model = load_model(sys.argv[1])
y = model.predict(x_data,batch_size=1200)
f = csv.writer(open(sys.argv[2],'w'), delimiter=',')
f.writerow(['id','label'])
for i in range(len(y)):
    f.writerow([i, y[i]])
