#!/bin/bash
x=$1
y=$2
tx=$3
dict=$4
wget 'https://www.dropbox.com/s/lz9fkfb4fw410b5/token.pickle?dl=1' -O token.pickle
wget 'https://www.dropbox.com/s/c503dlelard3l6n/wm.npy?dl=1' -O wm.npy
python segment.py $x $dict train.cut
python train6.py train.cut $y final-model.h5

