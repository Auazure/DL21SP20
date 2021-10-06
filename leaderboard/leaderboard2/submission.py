# Feel free to modifiy this file.

from torchvision import models, transforms
import torch.nn as nn

team_id = 20
team_name = "abc123"
email_address = "cj2164@nyu.edu"

class ft_model(nn.Module):
    def __init__(self, num_classes = 800):
        super(ft_model, self).__init__()
        self.pretrain = models.resnet50()
        self.pretrain.fc = nn.Linear(self.pretrain.fc.in_features, num_classes)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.relu(self.pretrain(x))
        outputs = self.linear(x)
        return outputs

def get_model():
    return ft_model(num_classes=800)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
