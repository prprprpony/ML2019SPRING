import numpy as np
import pandas as pd
import os, sys, time, datetime, pickle
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
# initlize model
model = load_model(sys.argv[2])
# preprocess data
x_data = pd.read_csv(os.popen("tail -n +2 " + sys.argv[1] + " | sed 's/,/ /g'"), sep=' ', header=None).values
y_data = to_categorical(x_data[:,0], 7)
x_data = np.delete(x_data,0,1) / 255
x_data = x_data.reshape((x_data.shape[0],48,48,1))
np.random.seed(20190409)
N = len(x_data)
p = np.random.permutation(N)
x_data = x_data[p]
y_data = y_data[p]
Ntest = N // 3
x_test, y_test = x_data[:Ntest], y_data[:Ntest]
x_train, y_train = x_data[Ntest:], y_data[Ntest:]
test_tot = np.sum(y_test,axis=0)
#print(test_tot)
train_stat = np.sum(y_train,axis=0)
#print(train_tot)
#print(test_tot / train_tot)

'''
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
'''
# training
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=100,epochs=100)
model.save(sys.argv[3])
pickle.dump(history.history,open(sys.argv[4],'wb'))
