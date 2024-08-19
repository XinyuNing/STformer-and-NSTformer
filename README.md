# STformer-and-NSTformer

## This program for checking of submission 12241 of AAAI 2025, contains the model, experimental environment and the experimental results of STformer and NSTformer.

## How to Run

### Download
```bash
    cd /path/to/your/project
    git clone 
```

### Run it

You can reproduce these models by running the following command:

```bash
    python experiments/train.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py --gpus '0'
```

Replace `${DATASET_NAME}` and `${MODEL_NAME}` with any supported models and datasets. For example, you can run NSTformer on METR-LA dataset by:

```bash
  python experiments/train.py -c baselines/NSTformer/METR-LA.py --gpus '0'
```

### Multiple GPUs

To run model on multiple GPUs, just modify "CFG.GPU_NUM" in dataset named file under model name, such as ".../NSTformer/METR-LA.py" then modify the gpu nums in above command, for example:

```bash
  python experiments/train.py -c baselines/NSTformer/METR-LA.py --gpus '0,1,2,3'
```



