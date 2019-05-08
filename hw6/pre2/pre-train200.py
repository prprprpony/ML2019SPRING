#!/usr/bin/env python3
import pickle
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
import gensim
import pandas as pd
import numpy as np

w2v = gensim.models.Word2Vec.load('word2vec.model')
print([x for x  in w2v.most_similar('dark',topn=100)])

t = Tokenizer()
vocab = w2v.wv.vocab
tot = [line for line in open('tot.cut','r')]
t.fit_on_texts(tot)
vsize = len(t.word_index) + 1
weight_matrix = np.zeros((vsize, w2v.vector_size))
print('w m',weight_matrix.shape)
print(t.word_index['四步'])
for w,i in t.word_index.items():
    #print(w)
    if i < 10:
        print(w,i)
    weight_matrix[i] = w2v[w]
emb = Embedding(vsize, output_dim=w2v.vector_size, weights=[weight_matrix], input_length=200, trainable=False)
pickle.dump(t, open('token200.pickle', 'wb'))
pickle.dump(emb, open('emb200.pickle', 'wb'))

