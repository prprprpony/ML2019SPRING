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
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
    model.add(Bidirectional(GRU(units=12, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(units=12, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dropout(0.3))
    model.add(Dense(units = 1, activation ='sigmoid'))
    rms = optimizers.RMSprop(clipnorm=1.)
    model.compile(optimizer = rms, loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(model.summary())
#exit(0)
# preprocess data
x_data = pre('train.cut')
y_data = pd.read_csv('train_y.csv')['label'].values
np.random.seed(20190507)
N = x_data.shape[0]
p = np.random.permutation(N)
x_data = x_data[p]
y_data = y_data[p]
Ntest = int(N * 0.2)
x_test, y_test = x_data[:Ntest], y_data[:Ntest]
x_train, y_train = x_data[Ntest:], y_data[Ntest:]
early_stopping=EarlyStopping(monitor='val_loss',patience=5)
best_weights_filepath = './best_weights.hdf5'
save_best_model=ModelCheckpoint(best_weights_filepath, monitor='val_loss', save_best_only=True, mode='auto')

try:
    model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[early_stopping, save_best_model], batch_size=1200,epochs=50)
except:
    model.save(datetime.datetime.now().strftime("%m-%d-%H:%M:%S") + '-CNN-crashed.h5')
model.load_weights(best_weights_filepath)
result = model.evaluate(x_test,y_test,batch_size=1200)
print(result)
accuracy = result[1]
model.save(datetime.datetime.now().strftime("%m-%d-%H:%M:%S-") + str(accuracy) + '-CNN.h5')
