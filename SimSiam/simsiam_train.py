import sys
import os
import argparse
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
# print(sys.path)

import torch
import torchvision
from torchvision import transforms
from PIL import Image # PIL is a library to process images
from torch import nn
import torch.nn.functional as F

from augmentations import get_aug
from simsiam import SimSiam

# from . import dataloader
from dataloader import CustomDataset

# from tqdm.notebook import tqdm





parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/')
parser.add_argument('--model-name', type=str, default='simsiam')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--net_size', type=int, default=50)
parser.add_argument('--temperature', type=int, default=1)

args = parser.parse_args()
checkpoint_path = args.checkpoint_dir + args.model_name

# sys.path.insert(1, args.checkpoint_dir)
# PATH = '/Users/colinwan/Desktop/NYU_MSDS/2572/FinalProject/DL21SP20'
PATH = ''
train_dataset = CustomDataset(root=PATH+'/dataset', split='unlabeled', transform=get_aug(train=True, image_size=96))
BATCH_SIZE = 256 
print(len(train_dataset))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model = SimSiam().to(device)
check = os.path.exists(
    os.path.join(checkpoint_path,
        args.model_name+"_encoder_{}.pth".format(args.net_size)))
print(os.path.join(checkpoint_path,
        args.model_name+"_encoder_{}.pth".format(args.net_size)))
print(check)

if check:
    print('Loading previous model')
    model.encoder.load_state_dict(torch.load(os.path.join(checkpoint_path,
        args.model_name+"_encoder_{}.pth".format(args.net_size))))
    model.projector.load_state_dict(torch.load(os.path.join(checkpoint_path,
        args.model_name+"_predictor_{}.pth".format(args.net_size))))



optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# print(p)

model.train()
EPOCHS=args.epochs

for epoch in range(EPOCHS):
    print('Current Epoch {}'.format(epoch))
    mean_loss = 0
    mean_acc = 0
    for idx, ((images1, images2), labels) in enumerate(train_dataloader):
        model.zero_grad()
        L = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
        loss = L.mean() # ddp
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()
        if idx%50==0:
          print('Loss: ', loss.item())
          torch.save(model.encoder.state_dict(), os.path.join(checkpoint_path,
            args.model_name+"_encoder_{}.pth".format(args.net_size)))
          torch.save(model.predictor.state_dict(), os.path.join(checkpoint_path, 
            args.model_name+"_predictor_{}.pth".format(args.net_size)))
    mean_loss /= len(train_dataloader)
    train_losses.append(mean_loss)
    print('Epoch Loss:', mean_loss)

    torch.save(model.encoder.state_dict(), os.path.join(checkpoint_path, 
        args.model_name+ "_encoder_{}.pth".format(args.net_size)))
    torch.save(model.predictor.state_dict(), os.path.join(checkpoint_path, 
        args.model_name+ "_predictor_{}.pth".format(args.net_size)))


print('Finish Training')
torch.save(model.encoder.state_dict(), os.path.join(checkpoint_path,
    args.model_name+ "_encoder_{}.pth".format(args.net_size)))
torch.save(model.projector.state_dict(), os.path.join(checkpoint_path, 
    args.model_name+"_predictor_{}.pth".format(args.net_size)))
print("Saved checkpoint to {os.path.join(args.checkpoint_dir, args.model_name, '.pth')}")





















