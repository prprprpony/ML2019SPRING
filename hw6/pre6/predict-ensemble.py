#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os, sys, csv
# preprocess data
MAX_LENGTH=30
def pre(fn):
    token = pickle.load(open('token.pickle','rb'))
    ret = [line for line in open(fn,'r')]
    ret = token.texts_to_sequences(ret)
    ret = pad_sequences(ret, maxlen=MAX_LENGTH)
    return ret

x_data = pre('test.cut')
y = np.zeros((x_data.shape[0],1))
num = 0
for m in sys.argv[1:-1]:
    print(m)
    num += 1
    model = load_model(m)
    ty = model.predict(x_data,batch_size=1200)
    y += ty
y /= num
f = csv.writer(open(sys.argv[-1],'w'), delimiter=',')
f.writerow(['id','label'])
for i in range(len(y)):
    f.writerow([i, int(y[i] > 0.5)])
