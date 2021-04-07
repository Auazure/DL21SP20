from torch import nn
import torch.nn.functional as F


class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.res_size = 512

        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.cl0 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2)
        self.cl1 = nn.Conv2d(in_channels=128, out_channels=self.res_size, kernel_size=3, stride=2)
        self.cl2 = nn.Conv2d(in_channels=self.res_size, out_channels=self.res_size, kernel_size=3, padding=1)
        self.cl3 = nn.Conv2d(in_channels=self.res_size, out_channels=self.res_size, kernel_size=3, padding=1)
        self.cl4 = nn.Conv2d(in_channels=self.res_size, out_channels=self.res_size, kernel_size=3, padding=1)

        self.cl8 = nn.Conv2d(in_channels=self.res_size, out_channels=self.res_size, kernel_size=3)

        self.bn1 = nn.BatchNorm2d(self.res_size)
        self.bn2 = nn.BatchNorm2d(self.res_size)
        self.bn3 = nn.BatchNorm2d(self.res_size)


        self.pre_block = nn.Sequential(self.cl0, self.relu,
                                       self.pool1,self.relu,
                                       self.cl1, self.pool2)
        self.resblock1 = nn.Sequential(self.cl2,
                                       self.pool2,
                                       self.bn1)
        self.resblock2 = nn.Sequential(self.cl3,
                                       self.pool2,
                                       self.bn2)
        self.resblock3 = nn.Sequential(self.cl4,
                                       self.pool2,
                                       self.bn3)

        self.final_block = nn.Sequential(self.cl8,
                                         self.relu,
                                         self.avgpool)


        self.ll1 = nn.Linear(in_features=8192, out_features=800)

        self.dense_layers = torch.nn.Sequential(self.ll1)


        self.output = F.log_softmax

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.pre_block(x)
        x = F.relu(x)
        x = self.resblock1(x)+x
        x = F.relu(x)
        x = self.resblock2(x)+x
        x = F.relu(x)
        x = self.resblock3(x)+x
        x = F.relu(x)

        x = self.final_block(x)

        x = F.relu(x)
        x = x.view(B,-1)
        x = self.dense_layers(x)
        x = self.output(x, dim=1)
        return x

