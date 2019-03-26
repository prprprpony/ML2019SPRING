import numpy as np
import pandas as pd
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
# preprocess data
x_data = pd.read_csv(os.popen("tail -n +2 train.csv | sed 's/,/ /g'"), sep=' ', header=None).values
y_data = to_categorical(x_data[:,0], 7)
x_data = np.delete(x_data,0,1) / 255
x_data = x_data.reshape((x_data.shape[0],48,48,1))
# initlize model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (48,48,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
for i in range(3):
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation ='relu'))
model.add(Dense(units = 7, activation ='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_data, y_data,batch_size=100,epochs=20)
print( model.evaluate(x_data,y_data) )
model.save('m1.h5')
