#!/bin/bash
wget 'https://www.dropbox.com/s/227ofca098zjjzu/final-model.h5?dl=1' -O final-model.h5
python saliency.py $1 $2
python filter.py $1 $2
