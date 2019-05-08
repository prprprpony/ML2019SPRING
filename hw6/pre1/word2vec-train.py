#!/usr/bin/env python3
from gensim.models import word2vec
sentences = word2vec.LineSentence('tot.cut')
model = word2vec.Word2Vec(sentences, size=300, min_count=1)
model.save('word2vec.model')
