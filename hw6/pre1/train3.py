#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import os, sys, time, datetime
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
from keras.layers import GRU,Bidirectional
from keras import optimizers
from keras.models import load_model
# initlize model
MAX_LENGTH=30
def pre(fn):
    token = pickle.load(open('token.pickle','rb'))
    ret = [line for line in open(fn,'r')]
    ret = token.texts_to_sequences(ret)
    ret = pad_sequences(ret, maxlen=MAX_LENGTH)
    return ret

emb = pickle.load(open('emb.pickle', 'rb'))
model = None
if len(sys.argv) == 3:
    print("load " + sys.argv[2] + " to keep training")
    model = load_model(sys.argv[2])
else:
    model = Sequential()
    model.add(emb)
    model.add(Bidirectional(GRU(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(units=128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dropout(0.3))
    model.add(Dense(units = 1, activation ='sigmoid'))
    rms = optimizers.RMSprop(clipnorm=1.)
    model.compile(optimizer = rms, loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(model.summary())
#exit(0)
# preprocess data
x_data = pre('train.cut')
y_data = pd.read_csv('train_y.csv')['label'].values
try:
    model.fit(x_data, y_data,batch_size=1200,epochs=50)
except:
    model.save(datetime.datetime.now().strftime("%m-%d-%H:%M:%S") + '-CNN-crashed.h5')
result = model.evaluate(x_data,y_data)
print(result)
accuracy = result[1]
model.save(datetime.datetime.now().strftime("%m-%d-%H:%M:%S-") + str(accuracy) + '-CNN.h5')
