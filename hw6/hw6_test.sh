#!/bin/bash
x=$1
dict=$2
out=$3
wget 'https://www.dropbox.com/s/cfglwk22kp7jmn8/rnn10-6.h5?dl=1' -O final-model.h5
wget 'https://www.dropbox.com/s/lz9fkfb4fw410b5/token.pickle?dl=1' -O token.pickle
python segment.py $x $dict x.cut
python predict.py final-model.h5 $out x.cut
