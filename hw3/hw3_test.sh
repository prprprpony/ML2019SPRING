#!/bin/bash
wget 'https://www.dropbox.com/s/227ofca098zjjzu/final-model.h5?dl=1' -O final-model.h5
python predict.py $1 $2 final-model.h5
