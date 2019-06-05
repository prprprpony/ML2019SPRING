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
model = load_model(sys.argv[1])
w = model.get_weights()
wo = []
for a in w:
    wo.append(a.flatten())
wo=np.concatenate(wo)
wo=wo.astype(np.float16)
f=open(sys.argv[2],'wb')
np.savez_compressed(f,wo)

