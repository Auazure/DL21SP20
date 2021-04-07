import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image 
from dataloader import CustomDataset
from baseline import CNN
import os
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
parser.add_argument('--epochs', type=int)
args = parser.parse_args()
sys.path.insert(1, args.checkpoint_dir)

# These numbers are mean and std values for channels of natural images.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Inverse transformation: needed for plotting.
unnormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

train_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(hue=.1, saturation=.1, contrast=.1),
    transforms.RandomRotation(20, resample=Image.BILINEAR),
    transforms.GaussianBlur(7, sigma=(0.1, 1.0)),
    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
    normalize,
])

validation_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    normalize,
])
# path = '/Users/colinwan/Desktop/NYU_MSDS/2572/FinalProject/DL21SP20'
path = ''
train_dataset = CustomDataset(root=path+'/dataset', split='train', transform=train_transforms)
validation_dataset = CustomDataset(root=path+'/dataset', split='val', transform=validation_transforms)
BATCH_SIZE = 128 

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=1)

from tqdm.notebook import tqdm

def get_loss_and_correct(model, batch, criterion, device):
    x, y = batch
    y_hat = model(x.to(device)).squeeze()
    y = y.to(torch.long).to(device)
    loss = criterion(y_hat,y)

    return loss, torch.sum(torch.argmax(y_hat,dim=1)==y)

def step(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


N_EPOCHS = args.epochs 

model = CNN()
if os.path.exists("baseline.pth"):
    model.load_state_dict(torch.load(args.checkpoint_dir +'/baseline.pth'))

criterion = nn.NLLLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
model.to(device)


train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []

pbar = tqdm(range(N_EPOCHS))

for i in pbar:
    print('Current Epoch .{}'.format(i))
    total_train_loss = 0.0
    total_train_correct = 0.0
    total_validation_loss = 0.0
    total_validation_correct = 0.0


    model.train()
    n=0
    for batch in tqdm(train_dataloader, leave=False):
        n+=1
        loss, correct = get_loss_and_correct(model, batch, criterion, device)
        step(loss, optimizer)
        total_train_loss += loss.item()
        total_train_correct += correct.item()

    with torch.no_grad():
        for batch in validation_dataloader:
            loss, correct = get_loss_and_correct(model, batch, criterion, device)
            total_validation_loss += loss.item()
            total_validation_correct += correct.item()

        
    mean_train_loss = total_train_loss / len(train_dataset)
    train_accuracy = total_train_correct / len(train_dataset)

    mean_validation_loss = total_validation_loss / len(validation_dataset)
    validation_accuracy = total_validation_correct / len(validation_dataset)
    print('Epoch: {}, Train Loss: {}, Vald Loss: {}'.format(i+1, mean_train_loss, mean_validation_loss))
    train_losses.append(mean_train_loss)
    validation_losses.append(mean_validation_loss)
    train_accuracies.append(train_accuracy)
    validation_accuracies.append(validation_accuracy)
    print('Epoch: {}, Train Accuracy: {}, Vald Accuracy: {}'.format(i+1, round(train_accuracy,3), round(validation_accuracy,3)))
    pbar.set_postfix({'train_loss': mean_train_loss, 'validation_loss': mean_validation_loss, 'train_accuracy': train_accuracy, 'validation_accuracy': validation_accuracy})
    
    
    
    
    
print('Finish Training')
os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'baseline.path'))
print("Saved checkpoint to {os.path.join(args.checkpoint_dir, 'baseline.path'}}")
      