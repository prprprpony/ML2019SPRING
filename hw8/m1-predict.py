#!/usr/bin/env python3
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,ELU,SeparableConv2D,GlobalAvgPool2D
from keras.optimizers import Adam
from keras.models import load_model
import os, sys, csv
# preprocess data
def m1():
    model = Sequential()

    model.add(Conv2D(15,(3,3),padding='same',input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(SeparableConv2D(15,(3,3),padding='same'))
    model.add(ELU())
    model.add(MaxPooling2D((2,2)))

    for f in [50,100]:
        model.add(Conv2D(f,(3,3),padding='same'))
        model.add(BatchNormalization())
        model.add(ELU())
        model.add(SeparableConv2D(f+20,(3,3),padding='same'))
        model.add(ELU())
        model.add(MaxPooling2D((2,2)))

    model.add(SeparableConv2D(120,(3,3),padding='same'))
    model.add(GlobalAvgPool2D())
    model.add(Dense(units=7,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer=Adam(0.0045),metrics=['accuracy'])
    return model

x_data = pd.read_csv(os.popen("tail -n +2 " + sys.argv[1] + " | sed 's/,/ /g'"), sep=' ', header=None).values
x_data = np.delete(x_data,0,1) / 255
x_data = x_data.reshape((x_data.shape[0],48,48,1))

N = x_data.shape[0]
model = m1()

weights_npy = np.load(sys.argv[3])['arr_0']
weights = []
ptr = 0
for empty_weight in model.get_weights():
    weight_len = len(empty_weight.ravel())
    weight = weights_npy[ptr:ptr + weight_len].reshape(empty_weight.shape)
    weights.append(weight.astype(np.float32))
    ptr += weight_len
assert ptr == len(weights_npy)
model.set_weights(weights)

y = model.predict_classes(x_data)
f = csv.writer(open(sys.argv[2],'w'), delimiter=',')
f.writerow(['id','label'])
for i in range(N):
    f.writerow([i, y[i]])
