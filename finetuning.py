from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from dataloader import CustomDataset
import argparse
import sys
import time
import os

parser = argparse.ArgumentParser()

# Input
parser.add_argument('--pretrained-dir-file', default='/checkpoints/barrowtwins/resnet50.pth', type=Path,
                    help='path and filename to pretrained model')

parser.add_argument('--model-name', default='resnet50', type=str,
                    help='pretrained model name')
parser.add_argument('--pretrained-algo', default='barrowtwins', type=str,
                    help='pretrained model method')
'''
If feature_extract = False, the model is finetuned and all model parameters are updated. 
If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
'''
parser.add_argument('--finetuning', default=True, type=bool,
                    help='finetune or feature extracting')

parser.add_argument('--image-size', default=96, type=int, metavar='N',
                    help='image size')
parser.add_argument('--num-classes', default=800, type=int, metavar='N',
                    help='num of class')

# For Training
parser.add_argument('--data', default='/dataset', type=Path,
                    help='path to dataset')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.01, type=float, metavar='LR',
                    help='base learning rate')

# For saving model and outputs
parser.add_argument('--checkpoint-dir', default='/checkpoints/finetuning', type=Path,
                    help='path to checkpoint directory')
parser.add_argument('--checkpoint-file', default='barrowtwins_resnet50.pth', type=str,
                    help='file name of checkpoint')

args = parser.parse_args()


def main():
    args = parser.parse_args()
    assert os.path.join(args.pretrained_dir_file)
    main_worker(0, args)


def main_worker(gpu, args):
    torch.cuda.set_device(gpu)
    device = torch.device('cuda')

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    validation_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(root=args.data, split='train', transform=train_transforms)
    validation_dataset = CustomDataset(root=args.data, split='val', transform=validation_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size,
                                                        num_workers=args.workers)

    # load pre-trained model from checkpoint
    model = pretrained_model(args.pretrained_algo, args.model_name, args.pretrained_dir_file, args.finetuning, args.num_classes)
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'finetuning_stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)
    best_validation_accuracy = 0
    since = time.time()

    for i in range(args.epochs):
        total_train_loss = 0.0
        total_train_correct = 0.0
        total_validation_loss = 0.0
        total_validation_correct = 0.0

        model.train()
        for batch in train_dataloader:
            loss, correct = get_loss_and_correct(model, batch, criterion, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
        # save the best model
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), args.checkpoint_dir / args.checkpoint_file)

        time_elapsed = time.time() - since
        print('Epoch: {}, Train Loss: {:.2f}, Val Loss: {:.2f}, Train Acc: {:.2f}, Val Acc: {:.2f}, Time: {}'.format(i,
                                                                                                                     mean_train_loss,
                                                                                                                     mean_validation_loss,
                                                                                                                     train_accuracy,
                                                                                                                     validation_accuracy,
                                                                                                                     time_elapsed))
        print('Epoch: {}, Train Loss: {:.2f}, Val Loss: {:.2f}, Train Acc: {:.2f}, Val Acc: {:.2f}, Time: {}'.format(i,
                                                                                                                     mean_train_loss,
                                                                                                                     mean_validation_loss,
                                                                                                                     train_accuracy,
                                                                                                                     validation_accuracy,
                                                                                                                     time_elapsed),
              file=stats_file)


def get_loss_and_correct(model, batch, criterion, device):
    x, y = batch
    x = x.to(device)
    y = y.to(torch.long).to(device)
    y_scores = model(x)
    loss = criterion(y_scores, y)
    return loss, torch.sum(torch.argmax(y_scores, dim=1) == y)


def set_parameter_requires_grad(model, finetuning):
    if not finetuning:
        for param in model.parameters():
            param.requires_grad = False


def pretrained_model(pretrained_algo, model_name, pretrained_dir_file, finetuning, num_classes):
    if model_name == "resnet18":
        """ Resnet18
        """
        if pretrained_algo == "torch":
            pre_model = torchvision.models.resnet18(pretrained=True)
            n_in_features = pre_model.fc.in_features
        else:
            pre_model = torchvision.models.resnet18()
            n_in_features = pre_model.fc.in_features
            if pretrained_algo == 'barrowtwins':
                pre_model.fc = nn.Identity()
            pre_model.load_state_dict(torch.load(pretrained_dir_file))

        set_parameter_requires_grad(pre_model, finetuning)
        pre_model.fc = nn.Linear(n_in_features, num_classes)

    elif model_name == "resnet50":
        """ Resnet50
        """
        if pretrained_algo == "torch":
            pre_model = torchvision.models.resnet50(pretrained=True)
            n_in_features = pre_model.fc.in_features
        else:
            pre_model = torchvision.models.resnet50()
            n_in_features = pre_model.fc.in_features
            if pretrained_algo == 'barrowtwins':
                pre_model.fc = nn.Identity()
            pre_model.load_state_dict(torch.load(pretrained_dir_file))

        set_parameter_requires_grad(pre_model, finetuning)
        pre_model.fc = nn.Linear(n_in_features, num_classes)


    else:
        print("Invalid model name, exiting...")
        exit()

print('Finish Training')

if __name__ == '__main__':
    main()
