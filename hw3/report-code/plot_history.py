import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import os, sys, time, datetime, pickle
# initlize model
history = pickle.load(open(sys.argv[1],'rb'))

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title(sys.argv[4] + ' Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(sys.argv[2])

# Plot training & validation loss values
plt.close()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title(sys.argv[4] + ' Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(sys.argv[3])

