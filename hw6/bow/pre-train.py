#!/usr/bin/env python3
import pickle
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
import gensim
import pandas as pd
import numpy as np

MAXV = 5000
t = Tokenizer(oov_token='UNK')
tot = [line for line in open('../tot.cut','r')]
freq = dict()
for line in tot:
    for w in line.split():
        if w in freq:
            freq[w] += 1
        else:
            freq[w] = 1
a = sorted([(v,k) for k,v in freq.items()],reverse=True)[:MAXV]
idx = dict()
for i in range(MAXV):
    idx[a[i][1]] = i
for i in range(len(tot)):
    tot[i] = ' '.join(w for w in tot[i].split() if w in idx)

t.fit_on_texts(tot)
vsize = len(t.word_index) + 1
print(vsize)
pickle.dump(t, open('token.pickle', 'wb'))
