#!/bin/bash
nohup python -u experiments/train.py -c baselines/NSTformer/METR-LA.py --gpus '0,1,2,3,6,7,8,9' > nstformer_metrla.out 2>&1 &
nohup python -u experiments/train.py -c baselines/NSTformer/PEMS-BAY.py --gpus '0,1,2,3,6,7,8,9' > nstformer_pemsbay.out 2>&1 &


