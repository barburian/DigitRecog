import torch as torch
import torchvision
import torchvision.transforms as transforms 
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import Compose
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120)
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2=nn.Conv2d( in_channels=6,out_channels=12,kernel_size=5)
        self.fc1=nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2= nn.Linear(in_features=120,out_features=60)
        self.out=nn.Linear(in_features=60,out_features=10)

    def forward(self,t):
        #implement later
        t=t
        #conv1 layer1
        t= self.conv1(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size=2,stride =2)
        #conv2 layer
        t=self.conv2(t)
        t=F.relu(t)
        t=F.max_pool2d(t,kernel_size=2,stride =2)
        #linear layer 1
        t=t.reshape(-1,12*4*4)
        t=self.fc1(t)
        t=F.relu(t)
        #linear layer 2
        t=self.fc2(t)
        t= F.relu(t)
        #output layer
        t=self.out(t)
        #t=F.softmax(t,dim=1)
        return t
network=Network()
train_set = torchvision.datasets.FashionMNIST(
    root = './data'
    ,train=True
    ,download=True,
    transform=Compose([transforms.ToTensor()])
)
sample=next(iter(train_set))
image,label=sample

output=network(image.unsqueeze(0))
print (output)
