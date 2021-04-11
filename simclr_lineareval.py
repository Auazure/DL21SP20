import torch
import torch.nn as nn
import torchvision
import numpy as np
from dataloader import CustomDataset
from simclr import SimCLR
from simclr.modules import LogisticRegression
from simclr.modules.transformations import TransformsSimCLR
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
parser.add_argument('--epochs', type=int)
args = parser.parse_args()


# path = '/Users/colinwan/Desktop/NYU_MSDS/2572/FinalProject/DL21SP20'
path = ''
train_dataset = CustomDataset(root=path + '/dataset', split='train', transform=TransformsSimCLR(96))
validation_dataset = CustomDataset(root=path + '/dataset', split='val', transform=TransformsSimCLR(96))
BATCH_SIZE = 128

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=1)


# from tqdm.notebook import tqdm

def get_loss_and_correct(model, batch, criterion, device):
    x, y = batch
    x = x.to(device)
    y = y.to(device)

    # get encoding
    with torch.no_grad():
        h, _, z, _ = simclr_model(x, x)
    h = h.detach()
    feature_vector = []
    labels_vector = []
    feature_vector.extend(h.cpu().detach().numpy())
    labels_vector.extend(y.numpy())
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)


    labels_vector_hat = model(feature_vector)
    loss = criterion(labels_vector_hat, labels_vector)


    return loss, torch.sum(torch.argmax(labels_vector_hat, dim=1) == labels_vector)


def step(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# load pre-trained model from checkpoint
assert os.path.join(args.checkpoint_dir, "simclr_encoder.pth")
encoder = torchvision.models.resnet50(pretrained=False)
simclr_model = SimCLR(encoder, 1024, 512)
simclr_model.to(device)
simclr_model.encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "simclr_encoder.pth")))
simclr_model.projector.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "simclr_projector.pth")))
simclr_model.eval()

N_EPOCHS = args.epochs

model = LogisticRegression(simclr_model.n_features, 128)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
model.to(device)

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []

# pbar = tqdm(range(N_EPOCHS))

for i in range(N_EPOCHS):
    print('Current Epoch .{}'.format(i))
    total_train_loss = 0.0
    total_train_correct = 0.0
    total_validation_loss = 0.0
    total_validation_correct = 0.0

    model.train()
    n = 0
    for batch in train_dataloader:
        n += 1
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
    print('Epoch: {}, Train Loss: {}, Vald Loss: {}'.format(i + 1, mean_train_loss, mean_validation_loss))
    train_losses.append(mean_train_loss)
    validation_losses.append(mean_validation_loss)
    train_accuracies.append(train_accuracy)
    validation_accuracies.append(validation_accuracy)
    print('Epoch: {}, Train Accuracy: {}, Vald Accuracy: {}'.format(i + 1, round(train_accuracy, 3),
                                                                    round(validation_accuracy, 3)))
#     pbar.set_postfix({'train_loss': mean_train_loss, 'validation_loss': mean_validation_loss, 'train_accuracy': train_accuracy, 'validation_accuracy': validation_accuracy})


print('Finish Training')
os.makedirs(args.checkpoint_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'simiclr_linear_eval.path'))
print("Saved checkpoint to {os.path.join(args.checkpoint_dir, 'simiclr_linear_eval.path'}}")
