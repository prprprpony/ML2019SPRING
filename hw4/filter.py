import numpy as np
import pandas as pd
import os, sys, time, datetime
import matplotlib.pyplot as plt

import keras.backend as K
from keras.utils import to_categorical
from keras.models import load_model
from scipy.misc import imsave
from keras.preprocessing.image import save_img

# preprocess data
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-6)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x,0,255).astype('uint8')
    return x
def _generate_filter_image(input_img,
                               layer_output,
                               filter_index,
                               input_img_data,epoch=20):

    loss = K.mean(layer_output[:,:,:,filter_index])

    grads = K.gradients(loss,input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-6)

    iterate = K.function([input_img], [loss, grads])

    for i in range(epoch):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * 1.
    return deprocess_image(input_img_data)

def _draw_filters(filters, img_name, output_dim=(48,48),n=None):
    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top n*n filters.
    # build a black picture with enough space for
    # e.g. our 8 x 8 filters of size 412 x 412, with a 5px margin in between
    print('len filters ',len(filters))
    if n is None:
        n = int(np.floor(np.sqrt(len(filters))))

    MARGIN = 5
    width = n * output_dim[0] + (n - 1) * MARGIN
    height = n * output_dim[1] + (n - 1) * MARGIN
    stitched_filters = np.zeros((width, height, 3), dtype='uint8')

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img = filters[i * n + j]
            width_margin = (output_dim[0] + MARGIN) * i
            height_margin = (output_dim[1] + MARGIN) * j
            stitched_filters[
                width_margin: width_margin + output_dim[0],
                height_margin: height_margin + output_dim[1], :] = img

    # save the result to disk
    save_img(img_name, stitched_filters)




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

model = load_model('final-model.h5')
assert len(model.inputs) == 1

layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = 'conv2d_1'
output_layer = layer_dict[layer_name]
input_img = model.inputs[0]
filter_lower = 0
filter_upper = 128

# iterate through each filter and generate its corresponding image
np.random.seed(20190412)
rnd_data = np.random.random((1,48,48,1))*20+128
f1 = [_generate_filter_image(input_img, output_layer.output, f, rnd_data) for f in range(filter_lower, filter_upper)]
f2 = [_generate_filter_image(input_img, output_layer.output, f, [x[0]], 10) for f in range(filter_lower, filter_upper)]

# Finally draw and store the best filters to disk
d = sys.argv[2]
if not os.path.exists(d):
    os.makedirs(d)
_draw_filters(f1, d+'/fig2_1.jpg')
_draw_filters(f2, d+'/fig2_2.jpg')
