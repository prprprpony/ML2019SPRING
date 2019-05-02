#!/bin/bash
time ./segment.py train_x.csv dict.txt.big train.cut
time ./segment.py test_x.csv dict.txt.big test.cut
cat train.cut test.cut > tot.cut
time ./word2vec-train.py
time ./pre-train.py
