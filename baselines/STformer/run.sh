#!/bin/bash
nohup python -u experiments/train.py -c baselines/STformer/METR-LA.py --gpus '0,1,2,3,6,7,8,9' > stformer_metrla.out 2>&1 &
nohup python -u experiments/train.py -c baselines/STformer/PEMS-BAY.py --gpus '0,1,2,3,6,7,8,9' > stformer_pemsbay.out 2>&1 &

