import numpy as np
import pandas as pd
import os, sys, time, datetime
import matplotlib.pyplot as plt

import keras.backend as K
from keras.utils import to_categorical
from keras.models import load_model
from lime import lime_image
import skimage

model = load_model('final-model.h5')
explainer = lime_image.LimeImageExplainer()
def predict(x):
    return model.predict(np.expand_dims(skimage.color.rgb2gray(x),3))
def segmentation(x):
    return skimage.segmentation.slic(x)
def explain(instance, predict_fn, **kwargs):
    np.random.seed(20190412)
    return explainer.explain_instance(instance, predict_fn, **kwargs)


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
    explaination = explain(
            x[i].squeeze(),
            predict,
            segmentation_fn=segmentation
            )
    image, mask = explaination.get_image_and_mask(
            label=i,
            positive_only=False,
            hide_rest=False,
            num_features=5,
            min_weight=0
            )
    plt.imsave(d + '/fig3_' + str(i) + '_ori.jpg', x[i].squeeze(), cmap=plt.cm.gray)
    plt.imsave(d + '/fig3_' + str(i) + '.jpg', image)


