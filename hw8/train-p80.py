#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os, sys, time, datetime
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,ELU,SeparableConv2D,GlobalAvgPool2D
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
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

model = None
if len(sys.argv) == 3:
    print("load " + sys.argv[2] + " to keep training")
    model = load_model(sys.argv[2])
else:
    model = m1()
    print(model.summary())
#exit(0)
# preprocess data
x_data = pd.read_csv(os.popen("tail -n +2 " + sys.argv[1] + " | sed 's/,/ /g'"), sep=' ', header=None).values
y_data = to_categorical(x_data[:,0], 7)
x_data = np.delete(x_data,0,1) / 255
x_data = x_data.reshape((-1,48,48,1))

N = len(x_data)
np.random.seed(20190606)
p = np.random.permutation(N)
x_data = x_data[p]
y_data = y_data[p]
Ntest = int(N * 0.2)
x_test, y_test = x_data[:Ntest], y_data[:Ntest]
x_train, y_train = x_data[Ntest:], y_data[Ntest:]

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

early_stopping=EarlyStopping(monitor='val_acc',patience=80)
best_weights_filepath = './best_weights.hdf5'
save_best_model=ModelCheckpoint(best_weights_filepath, monitor='val_acc', save_best_only=True, mode='auto')
# training
try:
    #model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[early_stopping, save_best_model], batch_size=100,epochs=150)
    model.fit_generator(
            datagen.flow(x_train, y_train,batch_size=100),
            validation_data=(x_test,y_test),
            callbacks=[early_stopping, save_best_model],
            steps_per_epoch=len(x_data)//100+1,epochs=404,workers=20)
except:
    model.save(datetime.datetime.now().strftime("%m-%d-%H:%M:%S") + '-m1-crashed.h5')

model.load_weights(best_weights_filepath)
result = model.evaluate(x_test,y_test,batch_size=100)
print(result)
accuracy = result[1]
val_loss = result[0]
model.save(datetime.datetime.now().strftime("%m-%d-%H:%M:%S-") + str(accuracy) + '-m1.h5')
