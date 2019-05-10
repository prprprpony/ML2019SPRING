#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd
import os, sys, time, datetime
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
from keras.layers import GRU,Bidirectional,Embedding
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

model = Sequential()
wm = np.load('wm.npy')
model.add(Embedding(wm.shape[0], output_dim=wm.shape[1], weights=[wm], input_length=MAX_LENGTH, trainable=False))
model.add(Bidirectional(GRU(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dropout(0.2))
model.add(Bidirectional(GRU(units=32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dropout(0.4))
model.add(Bidirectional(GRU(units=16, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dropout(0.4))
model.add(Bidirectional(GRU(units=8, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(GRU(units=4, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(GRU(units=2, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(units = 1, activation ='sigmoid'))
rms = optimizers.RMSprop(clipnorm=1.)
model.compile(optimizer = rms, loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model.summary())
#exit(0)
# preprocess data
x_data = pre('train.cut')
y_data = pd.read_csv('../train_y.csv')['label'].values
N = 119018
x_data = x_data[:N]
y_data = y_data[:N]
np.random.seed(20190507)
p = np.random.permutation(N)
x_data = x_data[p]
y_data = y_data[p]
Ntest = int(N * 0.2)
x_test, y_test = x_data[:Ntest], y_data[:Ntest]
x_train, y_train = x_data[Ntest:], y_data[Ntest:]
early_stopping=EarlyStopping(monitor='val_loss',patience=10)
best_weights_filepath = './best_weights.hdf5'
save_best_model=ModelCheckpoint(best_weights_filepath, monitor='val_loss', save_best_only=True, mode='auto')
name=sys.argv[1]
#history = model.fit(x_data,y_data,callbacks=[early_stopping, save_best_model], batch_size=1200,epochs=30)
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[save_best_model], batch_size=1200,epochs=39)
model.load_weights(best_weights_filepath)
result = model.evaluate(x_data,y_data,batch_size=1200)
print(result)
accuracy = result[1]
val_loss = result[0]
model.save(name)
pickle.dump(history.history,open(sys.argv[2],'wb'))
