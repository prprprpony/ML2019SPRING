import numpy as np
import pandas as pd
import os, sys, time
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
# initlize model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (48,48,1), activation = 'elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation = 'elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation = 'elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation = 'elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), activation = 'elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation = 'elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(units = 7, activation ='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())
#exit(0)
# preprocess data
x_data = pd.read_csv(os.popen("tail -n +2 " + sys.argv[1] + " | sed 's/,/ /g'"), sep=' ', header=None).values
y_data = to_categorical(x_data[:,0], 7)
x_data = np.delete(x_data,0,1) / 255
x_data = x_data.reshape((x_data.shape[0],48,48,1))
# training
try:
    model.fit(x_data, y_data,batch_size=100,epochs=60)
except:
    model.save(str(int(time.time())) + '-CNN-crashed.h5')
result = model.evaluate(x_data,y_data)
print(result)
accuracy = result[1]
model.save(str(accuracy) + '-' + str(int(time.time())) + '-CNN.h5')
