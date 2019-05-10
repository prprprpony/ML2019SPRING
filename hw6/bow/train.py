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

model = None
if len(sys.argv) == 2:
    print("load " + sys.argv[1] + " to keep training")
    model = load_model(sys.argv[1])
else:
    model = Sequential()
    model.add(Dense(units = 128, input_dim=MAXV+2,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = 64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units = 32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units = 16, activation='relu'))
    model.add(Dense(units = 8, activation='relu'))
    model.add(Dense(units = 4, activation='relu'))
    model.add(Dense(units = 1, activation ='sigmoid'))
    rms = optimizers.RMSprop(clipnorm=1.)
    model.compile(optimizer = rms, loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(model.summary())
#exit(0)
# preprocess data
x_data = pre('../train.cut')
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
best_weights_filepath = './best_weights.hdf5'
save_best_model=ModelCheckpoint(best_weights_filepath, monitor='val_loss', save_best_only=True, mode='auto')
name='bow.h5'
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[save_best_model], batch_size=1200,epochs=39)
model.load_weights(best_weights_filepath)
pickle.dump(history.history,open('bow.history','wb'))
result = model.evaluate(x_test,y_test,batch_size=1200)
print(result)
accuracy = result[1]
val_loss = result[0]
model.save(name)
