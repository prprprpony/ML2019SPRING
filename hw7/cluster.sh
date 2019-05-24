#!/bin/bash
img='images/'
tc='test_case.csv'
out='ans.csv'
if [[ $# -eq 3 ]]; then
  img=$1
  tc=$2
  out=$3
else 
  out=$1
fi

./cluster.py --image_dir $img --test_case $tc --output_name $out --load_state m1.pth
#    parser.add_argument('--epoch', default=100, type=int)
#    parser.add_argument('--batch', default=128, type=int)
#    parser.add_argument('--lr', default=5e-4, type=float)
#    parser.add_argument('--latent_dim', default=128, type=int)
