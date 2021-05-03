
# CoMatch
This is an  PyTorch implementation of [CoMatch: Semi-supervised Learning with Contrastive Graph Regularization](https://arxiv.org/abs/2011.11183).

The original source of PyTorch implementation is [here](https://github.com/salesforce/CoMatch).

## Usage

### Train
Train the model with the original label datasets and unlabeled dataset:

```
python Train_CoMatch.py --epochs <NEPOCH> --data <PATH of DATASET> --exp_dir <PATH of CHECKPOINTS>
```

### Monitoring training progress
```
%load_ext tensorboard
%tensorboard --logdir=<PATH of CHECKPOINTS>
```
