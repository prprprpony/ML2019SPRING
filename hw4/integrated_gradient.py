import numpy as np
import pandas as pd
import os, sys, time, datetime
import matplotlib.pyplot as plt

import keras.backend as K
from keras.utils import to_categorical
from keras.models import load_model

model = load_model('final-model.h5')

def get_saliency_map(image,i):
    x = np.expand_dims(image, axis=0)
    var_inputs = [model.input,K.learning_phase()]
    var_outputs = model.optimizer.get_gradients(model.output[0][i], model.input)
    get_gradients = K.function(inputs=var_inputs, outputs=var_outputs)
    return get_gradients([x,0])[0][0]


def get_map(image,i,base=None,ns=50):
    if base is None:
        base = np.zeros_like(image)
    diff = image - base
    grad = np.zeros_like(image)

    for alpha in np.linspace(0,1,ns):
        step = base + alpha * diff
        grad += get_saliency_map(step,i)

    return grad / ns * diff

# preprocess data
x_data = pd.read_csv(os.popen("tail -n +2 " + sys.argv[1] + " | sed 's/,/ /g'"), sep=' ', header=None).values
y_data = x_data[:,0]
x_data = np.delete(x_data,0,1) / 255
x_data = x_data.reshape((x_data.shape[0],48,48,1))

d = sys.argv[2]
if not os.path.exists(d):
    os.makedirs(d)
x = [None] * 7
w = 0
for i in range(len(x_data)):
    j = y_data[i]
    if x[j] is None:
        x[j] = x_data[i]
        w += 1
        if w == 7:
            break

for i in range(7):
    plt.imsave(d + '/fig4_' + str(i) + '_ori.jpg', x[i].squeeze(), cmap=plt.cm.gray)
    s = get_map(x[i],i).squeeze()
    plt.imshow(s, cmap=plt.cm.jet)
    plt.colorbar()
    plt.savefig(d + '/fig4_' + str(i) + '.jpg')
    plt.clf()


