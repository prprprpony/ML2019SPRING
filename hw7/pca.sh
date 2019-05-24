#!/bin/bash
img=$1
in=$2
re=$3
python3 pca/recon.py $img $in $re
