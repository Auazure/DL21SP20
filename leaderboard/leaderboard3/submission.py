# Feel free to modifiy this file.

from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math

team_id = 20
team_name = "abc123"
email_address = "cj2164@nyu.edu"


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, mlp=False, low_dim=128, in_channel=3, width=1, num_class=800):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool = nn.AvgPool2d(7, stride=1)

        self.classifier = nn.Linear(self.base * 8 * block.expansion, num_class)
        self.l2norm = Normalize(2)
        self.mlp = mlp
        if self.mlp:  # use an extra projection layer
            self.fc1 = nn.Linear(self.base * 8 * block.expansion, 2048)
            self.fc2 = nn.Linear(2048, low_dim)
        else:
            self.fc = nn.Linear(self.base * 8 * block.expansion, low_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat = x.view(x.size(0), -1)

        out = self.classifier(feat)

        if self.mlp:
            feat = F.relu(self.fc1(feat))
            feat = self.fc2(feat)
        else:
            feat = self.fc(feat)
        feat = self.l2norm(feat)
        return out, feat


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


class Model(nn.Module):

    def __init__(self, base_encoder, args, width):
        super(Model, self).__init__()

        self.K = args.K

        self.encoder = base_encoder(num_class=args.num_class, mlp=True, low_dim=args.low_dim, width=width)
        self.m_encoder = base_encoder(num_class=args.num_class, mlp=True, low_dim=args.low_dim, width=width)

        for param, param_m in zip(self.encoder.parameters(), self.m_encoder.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        # queue to store momentum feature for strong augmentations
        self.register_buffer("queue_s", torch.randn(args.low_dim, self.K))
        self.queue_s = F.normalize(self.queue_s, dim=0)
        self.register_buffer("queue_ptr_s", torch.zeros(1, dtype=torch.long))
        # queue to store momentum probs for weak augmentations (unlabeled)
        self.register_buffer("probs_u", torch.zeros(args.num_class, self.K))

        # queue (memory bank) to store momentum feature and probs for weak augmentations (labeled and unlabeled)
        self.register_buffer("queue_w", torch.randn(args.low_dim, self.K))
        self.register_buffer("queue_ptr_w", torch.zeros(1, dtype=torch.long))
        self.register_buffer("probs_xu", torch.zeros(args.num_class, self.K))

        # for distribution alignment
        self.hist_prob = []

    def forward(self, x):
        x = x.cuda()
        outputs_x, _ = self.encoder(x)
        return outputs_x


parser = argparse.ArgumentParser(description='CoMatch Evaluation')
parser.add_argument('--low-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--K', default=30000, type=int, help='size of memory bank and momentum queue')
parser.add_argument('--num-class', default=800, type=int)
args, unknown = parser.parse_known_args()


def get_model():
    model = Model(resnet50, args, 1)
    return model


eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])