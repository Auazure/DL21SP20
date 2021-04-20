import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import CustomDataset
from FM.dataset.cifar import TransformFixMatch,DATASET_GETTERS
from tqdm.notebook import tqdm
import argparse
from torchvision.models import resnet18, resnet50

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/')
parser.add_argument('--model-name', type=str, default='FixMatch')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--net-size', type=int, default=50)
parser.add_argument('--temperature', type=int, default=1)
args = parser.parse_args()


checkpoint_path = args.checkpoint_dir + args.model_name

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=96,
                              padding=int(96*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
transform_unlabled = TransformFixMatch(mean=mean, std=std, size=96)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Inverse transformation: needed for plotting.
unnormalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)
validation_transforms = transforms.Compose([
                                    transforms.Resize((96, 96)),
                                    transforms.ToTensor(), 
                                    normalize,
                                ])
BATCH_SIZE = 128 # TODO
mu = 4
PATH = ''
# PATH = '/Users/colinwan/Desktop/NYU_MSDS/2572/FinalProject/DL21SP20'

unlabeled_dataset = CustomDataset(PATH+'/dataset', 'unlabeled', transform=transform_unlabled)
unlabeled_trainloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE*mu, shuffle=True, num_workers=1)

labeled_dataset = CustomDataset(PATH+'/dataset', 'train', transform=transform_labeled)
labeled_trainloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

validation_dataset = CustomDataset(PATH+'/dataset', 'val', validation_transforms)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=1)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


model = resnet18() if args.net_size=='18' else resnet50()
model.fc = nn.Linear(512, 800)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.03,momentum=0.9, nesterov=True)

model_param_path = os.path.join(checkpoint_path,
        args.model_name+"_{}_checkpoint.pt".format(args.net_size))
check = os.path.exists(model_param_path)
print(model_param_path)
print(check)

if check:
    print('Loading previous model')
    checkpoint = torch.load(model_param_path, map_location=device)
    model.load_state_dict(checkpoint['model_resnet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



labeled_iter = iter(labeled_trainloader)
unlabeled_iter = iter(unlabeled_trainloader)
eval_step = 1000
EPOCHS = args.epochs
temperature = args.temperature
lambda_un = 1
period = 300
model.train()


for epoch in range(EPOCHS):
	print('Current Epoch {}'.format(epoch))
	mean_l_loss = 0
	mean_u_loss = 0
	mean_l_acc = 0
	mean_u_acc = 0

	cur_l_loss = 0
	cur_u_loss = 0
	cur_l_acc = 0
	cur_u_acc = 0
	for idx in range(eval_step):
		# Get Data
		model.zero_grad()
		try:
		    inputs_x, targets_x = labeled_iter.next()
		except:
		    labeled_iter = iter(labeled_trainloader)
		    inputs_x, targets_x = labeled_iter.next()
		targets_x = targets_x.to(device)
		try:
		    (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
		except:
		    unlabeled_iter = iter(unlabeled_trainloader)
		    (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()


		batch_size = inputs_x.shape[0]

		inputs = torch.cat([inputs_x, inputs_u_w, inputs_u_s]).to(device)
		logits = model(inputs)

		logits_labeled = logits[:batch_size].to(device)
		logits_u_w = logits[batch_size:2*batch_size].to(device)
		logits_u_s = logits[batch_size*2:].to(device)

		Lx = F.cross_entropy(logits_labeled, targets_x, reduction='mean')
		pseudo_label = torch.softmax(logits_u_w.detach()/temperature, dim=-1).to(device)
		max_probs, targets_u = torch.max(pseudo_label, dim=-1)
		mask = max_probs.ge(0.95).float()
		Lu = (F.cross_entropy(logits_u_s, targets_u,
		                      reduction='none') * mask).mean()

		loss = Lx + lambda_un * Lu
		loss.backward()
		optimizer.step()

		_, y_l_labeled = torch.max(torch.softmax(logits_labeled.detach()/temperature, dim=-1), dim=-1)
		_, y_u_labeled = torch.max(torch.softmax(logits_u_s.detach()/temperature, dim=-1), dim=-1)

		l_acc = (y_l_labeled==targets_x.to(device)).sum()/batch_size
		u_acc = (y_u_labeled==targets_u.to(device)).sum()/batch_size/mu
		mean_l_loss += Lx.item()
		mean_u_loss += Lu.item()
		mean_l_acc += l_acc.item()
		mean_u_acc += u_acc.item()

		cur_l_loss += Lx.item()
		cur_u_loss += Lu.item()
		cur_l_acc += l_acc.item()
		cur_u_acc += u_acc.item()

		if idx%period==0:
			print('---------------')
			if idx >= period:
				print('Label Loss: {}, Unlabel Loss: {}, Label Acc: {}, Unlabel Acc: {}'.
					format(cur_l_loss/period, cur_u_loss/period, cur_l_acc/period, cur_u_acc/period))
			# print('Label Loss: ', cur_l_loss/period)
			# print('Unlabel Loss: ', cur_u_loss/period)
			# print('Label Acc: ', cur_l_acc/period)
			# print('Unlabel Acc: ', cur_u_acc/period)
			else:
				print('Label Loss: {}, Unlabel Loss: {}, Label Acc: {}, Unlabel Acc: {}'.
					format(cur_l_loss, cur_u_loss, cur_l_acc, cur_u_acc))

			# print('Label Loss: ', cur_l_loss)
			# print('Unlabel Loss: ', cur_u_loss)
			# print('Label Acc: ', cur_l_acc)
			# print('Unlabel Acc: ', cur_u_acc)
			print('---------------')

			cur_l_loss =0
			cur_u_loss =0
			cur_l_acc =0
			cur_u_acc =0
			# if idx >=period:
			torch.save({
				'model_resnet_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				}, model_param_path)

	mean_l_loss /= eval_step
	mean_u_loss /= eval_step
	mean_l_acc /= eval_step
	mean_u_acc /= eval_step

	val_loss = 0
	val_acc = 0
	with torch.no_grad():
		for batch in validation_dataloader:
			inputs_x, targets_x = batch
			logits = model(inputs_x.to(device))
			loss = F.cross_entropy(logits, targets_x.to(device), reduction='mean')

			_, yhat = torch.max(torch.softmax(logits.detach(), dim=-1), dim=-1)
			correct = (yhat==targets_x.to(device)).sum()/batch_size

			val_loss += loss.item()
			val_acc += correct.item()
	val_final_loss = val_loss/len(validation_dataloader)    
	val_final_acc = val_acc/len(validation_dataloader)   

	print('###############')
	print('###############')
	print('Epoch Info:')
	print('Label Loss: {}, Unlabel Loss: {}, Label Acc: {}, Unlabel Acc: {}'.
	      		format(mean_l_loss, mean_u_loss, mean_l_acc, mean_u_acc))
	        
	# print('Label Loss: ', mean_l_loss)
	# print('Unlabel Loss: ', mean_u_loss)
	# print('Label Acc: ', mean_l_acc)
	# print('Unlabel Acc: ', mean_u_acc)
	print('---------------')
	print('Val Info')
	print('Epoch Val Loss: {}, Epoch Val Acc: {}'.format(val_final_loss, val_final_acc))
	print('###############')
	print('###############')

















