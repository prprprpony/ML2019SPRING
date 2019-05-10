#!/usr/bin/env python3
from gensim.models import word2vec
sentences = word2vec.LineSentence('tot.cut')
model = word2vec.Word2Vec(sentences, size=300, iter=35, sg=1, workers=20)
model.save('word2vec.model')
