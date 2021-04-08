from torch import nn
import torch.nn.functional as F
import torch


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
        self.cl2 = nn.Conv2d(in_channels=self.res_size, out_channels=self.res_size, kernel_size=3)

        self.pre_block = nn.Sequential(self.cl0, self.relu,
                                       self.pool1,self.relu,
                                       self.cl1, self.pool2)
        
        self.resblock1 = self._makeresblcok()
        self.resblock2 = self._makeresblcok()
        self.resblock3 = self._makeresblcok()
        self.resblock4 = self._makeresblcok()
        self.resblock5 = self._makeresblcok()
        self.resblock6 = self._makeresblcok()

        self.final_block = nn.Sequential(self.cl2,
                                         self.relu,
                                         self.avgpool)


        self.ll1 = nn.Linear(in_features=8192, out_features=800)

        self.dense_layers = torch.nn.Sequential(self.ll1)


        self.output = F.log_softmax

    def _makeresblcok(self):
        cnn = nn.Conv2d(in_channels=self.res_size, out_channels=self.res_size, kernel_size=3, padding=1)
        pool = self.pool2
        norm = nn.BatchNorm2d(self.res_size)
        seq = nn.Sequential(cnn, pool, norm)
        return seq        
        
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
        x = self.resblock4(x)+x
        x = F.relu(x)
        x = self.resblock5(x)+x
        x = F.relu(x)
        x = self.resblock6(x)+x
        x = F.relu(x)
        x = self.final_block(x)

        x = F.relu(x)
        x = x.view(B,-1)
        x = self.dense_layers(x)
        x = self.output(x, dim=1)
        return x

