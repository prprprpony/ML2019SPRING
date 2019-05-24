#!/bin/bash
if [[ $# -eq 1 ]]; then
  ./cluster.py --image_dir images/  --test_case test_case.csv --model_name $1 --epoch 50
else
  ./cluster.py --image_dir images/  --test_case test_case.csv --model_name $1 --load_state $2
fi

#    parser.add_argument('--epoch', default=100, type=int)
#    parser.add_argument('--batch', default=128, type=int)
#    parser.add_argument('--lr', default=5e-4, type=float)
#    parser.add_argument('--latent_dim', default=128, type=int)
