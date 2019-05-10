#!/usr/bin/env python3
import pickle
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
import gensim
import pandas as pd
import numpy as np

w2v = gensim.models.Word2Vec.load('word2vec.model')

t = Tokenizer(oov_token='UNK')
vocab = w2v.wv.vocab
tot = [line for line in open('tot.cut','r')]
for i in range(len(tot)):
    tot[i] = ' '.join(w for w in tot[i].split() if w in w2v)
print(tot[:10])
t.fit_on_texts(tot)
vsize = len(t.word_index) + 1
weight_matrix = np.zeros((vsize, w2v.vector_size))
for w,i in t.word_index.items():
    if w in w2v:
        weight_matrix[i] = w2v[w]
np.save('wm.npy',weight_matrix)
pickle.dump(t, open('token.pickle', 'wb'))

