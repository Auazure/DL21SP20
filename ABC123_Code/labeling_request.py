from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from dataloader import CustomDataset
import pandas as pd
import numpy as np
import argparse
import sys
import time
import os

parser = argparse.ArgumentParser()

# Input
parser.add_argument('--checkpoint-dir', default='./checkpoints/finetuning', type=Path,
                    help='path to model')
parser.add_argument('--checkpoint-file', default='./checkpoints/barlowtwins/barlowtwins_resnet50_epochs100.pth', type = str,
                    help='filename of model')

parser.add_argument('--model-name', default='resnet50', type=str,
                    help='pretrained model name')
parser.add_argument('--pretrained-algo', default='barlowtwins', type=str,
                    help='pretrained model method')

parser.add_argument('--data',  default='/dataset', type=Path,
                    help='path to dataset')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loader workers') # Real Default 8
parser.add_argument('--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size')

args = parser.parse_args()

def main_worker(gpu, args):
    torch.cuda.set_device(gpu)
    device = torch.device('cuda')


    basic_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    unlabeled_dataset = CustomDataset(root=args.data, split='unlabeled', transform=basic_transforms)
    labeled_dataset = CustomDataset(root=args.data, split='train', transform=basic_transforms)

    unlabeled_dataset = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=args.batch_size, num_workers=args.workers)
    labeled_dataset = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size,
                                                        num_workers=args.workers)

    # load pre-trained model from checkpoint
    model = ft_model()
    model.load_state_dict(torch.load(args.checkpoint_dir/args.checkpoint_file))
    model.eval()
    model.to(device)

    unlabeled_entropy = []
    labeled_entropy = []
    since = time.time()
    steps = 100
    with torch.no_grad():
        for i, batch in enumerate(unlabeled_dataset):
            entropy = get_entropy(model, batch, device).tolist()
            unlabeled_entropy.extend(entropy)
            if i%steps == 0:
                print(i, sum(entropy)/len(entropy))
        for i, batch in enumerate(labeled_dataset):
            entropy = get_entropy(model, batch, device).tolist()
            labeled_entropy.extend(entropy)
            if i%steps == 0:
                print(i, sum(entropy)/len(entropy))
    return unlabeled_entropy, labeled_entropy

def get_entropy(model, batch, device):
    x, _ = batch
    x = x.to(device)
    y_scores = model(x)
    entropy = cal_entropy(y_scores)
    return entropy

def cal_entropy(x):
    e = nn.functional.softmax(x, dim=1) * nn.functional.log_softmax(x, dim=1)
    e = -1.0 * e.sum(dim=1)
    return e

class ft_model(nn.Module):
    def __init__(self, num_classes = 800):
        super(ft_model, self).__init__()
        self.pretrain = torchvision.models.resnet50()
        self.pretrain.fc = nn.Linear(self.pretrain.fc.in_features, num_classes)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.relu(self.pretrain(x))
        outputs = self.linear(x)
        return outputs

unlabeled_entropy, labeled_entropy = main_worker(0, args)
unlabeled_entropy, labeled_entropy = np.array(unlabeled_entropy), np.array(labeled_entropy)
max_unlabeled_entropy, max_labeled_entropy = unlabeled_entropy.argsort()[::-1], labeled_entropy.argsort()[::-1]
max_unlabeled_entropy_selected = max_unlabeled_entropy[:12800]
# max_labeled_entropy_selected = max_labeled_entropy[:4000]
max_unlabeled_entropy_selected.sort()
request_20 = pd.Series(max_unlabeled_entropy_selected).map(lambda x: str(x)+".png")
with open('./request_20.csv','w') as f:
    for l in request_20:
         f.write(l+',\n')