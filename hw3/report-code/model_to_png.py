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
model = load_model(sys.argv[1])
plot_model(model, to_file=sys.argv[2], show_shapes=True, show_layer_names=True, dpi=200)
