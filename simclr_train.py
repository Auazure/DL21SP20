from torch import nn
import torch.nn.functional as F
import torchvision
import torch

from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model
from baseline import CNN
from dataloader import CustomDataset

import os
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
parser.add_argument('--epochs', type=int)
parser.add_argument('--temperature', type=int, default=1)

args = parser.parse_args()
# sys.path.insert(1, args.checkpoint_dir)
path = ''

train_dataset = CustomDataset(root=path+'/dataset', split='unlabeled', transform=TransformsSimCLR(96))
BATCH_SIZE = 256 

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

def train(train_loader, model, criterion, optimizer, args):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()
        if step%100==0:
            print('Step: {}, Train Loss: {}'.format(step, loss.item()))
#             os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.encoder.state_dict(), os.path.join(args.checkpoint_dir, 'simclr_encoder.path'))
            torch.save(model.projector.state_dict(), os.path.join(args.checkpoint_dir, 'simclr_projector.path'))
        loss_epoch += loss.item()
    return loss_epoch

encoder = torchvision.models.resnet50(pretrained=False)
criterion = NT_Xent(BATCH_SIZE, args.temperature, 1)
model = SimCLR(encoder, 256, 2048)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

if torch.cuda.is_available():
    criterion = criterion.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
p = args.checkpoint_dir+"/simclr.pth"
print(p)
check = os.path.exists(args.checkpoint_dir+"/simclr.pth")
print(check)

if os.path.exists(args.checkpoint_dir+"/simclr.pth"):
    print('Loading previous model')
    model.load_state_dict(torch.load(args.checkpoint_dir +'/simclr.pth'))

model = model.to(device)

model.train()
EPOCHS=10
for i in range(EPOCHS):
    print('Current Epoch .{}'.format(i))
    total_train_loss = 0.0
    total_train_correct = 0.0
    total_validation_loss = 0.0
    total_validation_correct = 0.0
    
    
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_dataloader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()
        if step%100==0:
            print('Step: {}, Train Loss: {}'.format(step, loss.item()))
#             os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.encoder.state_dict(), os.path.join(args.checkpoint_dir, 'simclr_encoder.path'))
            torch.save(model.projector.state_dict(), os.path.join(args.checkpoint_dir, 'simclr_projector.path'))
        loss_epoch += loss.item()
    
    
    
    
    
    
#     loss_epoch = train(train_dataloader, model, criterion, optimizer, args)
    avg_loss = loss_epoch/len(train_dataloader)
    
    print('Epoch: {}, Train Loss: {}'.format(i+1, avg_loss))



print('Finish Training')
# os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(model.encoder.state_dict(), os.path.join(args.checkpoint_dir, 'simclr_encoder.path'))
torch.save(model.projector.state_dict(), os.path.join(args.checkpoint_dir, 'simclr_projector.path'))
print("Saved checkpoint to {os.path.join(args.checkpoint_dir, 'simclr.path')}")





















