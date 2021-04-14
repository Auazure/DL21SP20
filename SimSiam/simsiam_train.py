import torch
import torchvision
from torchvision import transforms
from PIL import Image # PIL is a library to process images
from torch import nn
import torch.nn.functional as F

from augmentations import get_aug
from simsiam import SimSiam

from dataloader import CustomDataset

from dataloader import CustomDataset
from tqdm.notebook import tqdm

import os
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str, default='')
parser.add_argument('--model-name', type=str, default='/simsiam')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--temperature', type=int, default=1)

args = parser.parse_args()

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
if os.path.exists(PATH+'/simsiam_encoder_18.pth'):
  model.encoder.load_state_dict(torch.load(PATH+'/simsiam_encoder_18.pth'))
  model.predictor.load_state_dict(torch.load(PATH+'/simsiam_predictor_18.pth'))
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# print(p)
check = os.path.exists(args.checkpoint_dir+args.model_name+"_encoder_18.pth")
print(check)

if check:
    print('Loading previous model')
    model.encoder.load_state_dict(torch.load(args.checkpoint_dir +args.model_name+"_encoder_18.pth"))
    model.projector.load_state_dict(torch.load(args.checkpoint_dir +args.model_name+"_predictor_18.pth"))


model.train()
EPOCHS=args.epochs

for epoch in tqdm(range(EPOCHS)):   
    print('Current Epoch {}'.format(epoch))
    mean_loss = 0
    mean_acc = 0
    for idx, ((images1, images2), labels) in enumerate(tqdm(train_dataloader, leave=False)):
        model.zero_grad()
        L = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))
        loss = L.mean() # ddp
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()
        if idx%50==0:
          print('Loss: ', loss.item())
          torch.save(model.encoder.state_dict(), os.path.join(PATH, args.model_name+ "_encoder_18.pth"))
          torch.save(model.predictor.state_dict(), os.path.join(PATH, args.model_name+ "_predictor_18.pth"))
    mean_loss /= len(train_dataloader)
    train_losses.append(mean_loss)
    print('Epoch Loss:', mean_loss)

    torch.save(model.encoder.state_dict(), os.path.join(PATH, args.model_name+ "_encoder_18.pth"))
    torch.save(model.predictor.state_dict(), os.path.join(PATH, args.model_name+ "_predictor_18.pth"))


print('Finish Training')
torch.save(model.encoder.state_dict(), os.path.join(args.checkpoint_dir,args.model_name+ "_encoder_18.pth"))
torch.save(model.projector.state_dict(), os.path.join(args.checkpoint_dir, args.model_name+"_predictor_18.pth"))
print("Saved checkpoint to {os.path.join(args.checkpoint_dir, args.model_name, '.pth')}")





















