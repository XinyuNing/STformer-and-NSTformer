#!/bin/bash
nohup python -u experiments/train.py -c baselines/oN/METR-LA.py --gpus '0,1,2,3,6,7,8,9' > on_metrla.out 2>&1 &
nohup python -u experiments/train.py -c baselines/oN/PEMS-BAY.py --gpus '0,1,2,3,6,7,8,9' > on_pemsbay.out 2>&1 &

