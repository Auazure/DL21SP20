
# CoMatch
This is an  PyTorch implementation of [CoMatch: Semi-supervised Learning with Contrastive Graph Regularization](https://arxiv.org/abs/2011.11183).
The official PyTorch implementation is [here](https://github.com/salesforce/CoMatch).

## Usage

### Train
Train the model with the original label datasets and unlabeled dataset:

```
python Train_CoMatch.py --epochs <NEPOCH> --data <PATH of DATASET> --exp_dir <PATH of CHECKPOINTS>
```
Before training with extra label, make sure to copy `label_20.pt` and `img_idx.pt` into the dataset folder.
Train the model with the original and extra label datasets and unlabeled dataset:

```
python Train_CoMatch_New.py --epochs <NEPOCH> --data <PATH of DATASET> --exp_dir <PATH of CHECKPOINTS> --resume <PATH of PREVIOUS MODEL PARAM>
```
If one wishes to train on Greene, adjust the code in `train.sbatch` accordingly and run
```
sbatch train.sbatch
```
### Monitoring training progress
```
%load_ext tensorboard
%tensorboard --logdir=<PATH of CHECKPOINTS>
```

# Labeling Request
Note, when we submitted our labeling request, we haven't gotten outstanding CoMatch outcome. Hence, the labeling is based on Barlow Twins.
## Barlow-Twins
The original source of PyTorch implementation is [here](https://github.com/facebookresearch/barlowtwins).

### 1) Pre-trained 
Train the Barlow-Twins model with unlabeled dataset. 
* You can monitor the training progress through "stats_50.txt" file produced by this code.
* You will get "resnet50.pth", "checkpoint_50.pth" after the successfully executed the code. 
```
sbatch barlowtwins_train.sbatch
```

### 2) Fine-tune 
Fine-tune the Barlow-Twins pretrained model with the original label datasets. 
* You can monitor the training progress through "finetuned_barlowtwins_resnet50_epochs100_stats.txt" file produced by this code.
* You will get "finetuned_barlowtwins_resnet50_epochs100.pth" after the successfully executed the code.  
```
sbatch finetuning.sbatch
```

## Labeling images
Get 12800 images which should be labeled through calculating entropy through fine-tuned Barlow-Twins model. 
```
sbatch labeling_request.sbatch
```

