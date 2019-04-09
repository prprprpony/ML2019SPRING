import numpy as np
import pandas as pd
import os, sys, time, datetime
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
# initlize model
model = Sequential()
model.add(Flatten(input_shape=(48,48,1)))
model.add(Dense(units = 512, activation ='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units = 512, activation ='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(units = 512, activation ='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units = 666, activation ='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(units = 1024, activation ='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(units = 1024, activation ='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(units = 1024, activation ='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(units = 1024, activation ='elu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(units = 7, activation ='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())
model.save('dnn.h5')
exit(0)
# preprocess data
x_data = pd.read_csv(os.popen("tail -n +2 " + sys.argv[1] + " | sed 's/,/ /g'"), sep=' ', header=None).values
y_data = to_categorical(x_data[:,0], 7)
x_data = np.delete(x_data,0,1) / 255
x_data = x_data.reshape((x_data.shape[0],48,48,1))
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
# training
try:
    #model.fit(x_data, y_data,batch_size=100,epochs=150)
    model.fit_generator(datagen.flow(x_data, y_data,batch_size=100),steps_per_epoch=len(x_data)*2//100,epochs=505,workers=10)
except:
    model.save(datetime.datetime.now().strftime("%m-%d-%H:%M:%S") + '-CNN7-crashed.h5')
result = model.evaluate(x_data,y_data)
print(result)
accuracy = result[1]
