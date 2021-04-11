import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from dataloader import CustomDataset
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str)
parser.add_argument('--epochs', type=int)
args = parser.parse_args()

train_transforms = transforms.Compose([
    transforms.ToTensor(),  # convert PIL to Pytorch Tensor
])

validation_transforms = transforms.Compose([
    transforms.ToTensor(),
])

path = ''
# train_dataset = CustomDataset(root=path, split='train', transform=train_transforms)
# validation_dataset = CustomDataset(root=path, split='val', transform=validation_transforms)

train_dataset = CustomDataset(root=path + '/dataset', split='train', transform=train_transforms)
validation_dataset = CustomDataset(root=path + '/dataset', split='val', transform=validation_transforms)
BATCH_SIZE = 128

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=1)


# from tqdm.notebook import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# load pre-trained model from checkpoint
# assert os.path.join(args.checkpoint_dir, "simclr_encoder.pth")
encoder = torchvision.models.resnet18(pretrained=True)
encoder.to(device)
# encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "simclr_encoder.pth")))

# N_EPOCHS = 1
N_EPOCHS = args.epochs

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

model = LogisticRegression(1000, 800)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
model.to(device)

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []


def get_loss_and_correct(encoder, model, batch, criterion, device):
    x, y = batch
    x = x.to(device)
    y = y.to(torch.long).to(device)

    # get encoding
    with torch.no_grad():
        h = encoder(x)
    h = h.detach()
    y_scores = model(h)
    print(y_scores)
    print(y)
    loss = criterion(y_scores, y)
    return loss, torch.sum(torch.argmax(y_scores, dim=1) == y)

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
        loss, correct = get_loss_and_correct(encoder, model, batch, criterion, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        total_train_correct += correct.item()

    with torch.no_grad():
        for batch in validation_dataloader:
            loss, correct = get_loss_and_correct(encoder, model, batch, criterion, device)
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