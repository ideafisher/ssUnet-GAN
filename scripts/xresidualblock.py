
import torch
import torch.nn as nn
### models ###
class Gaussian(nn.Module):
    def forward(self,input):
        return torch.exp(-torch.mul(input,input))

class Modulecell(nn.Module):
    def __init__(self,in_channels=1,out_channels=64,kernel_size=3,skernel_size=9):
        super(Modulecell,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=((kernel_size-1)//2),bias=True))
        self.module = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=skernel_size,stride=1,padding=((skernel_size-1)//2),groups=out_channels),
            nn.BatchNorm2d(out_channels),
            Gaussian())
    def forward(self,x):
        x1 = self.features(x)
        x2 = self.module(x1)
        x = torch.mul(x1,x2)
        return x

class xResidualBlock(nn.Module):
    def __init__(self,in_channels=64,planes=64,kernel_size=3, s=1):
        super(xResidualBlock,self).__init__()
        self.md = Modulecell(in_channels,planes,kernel_size)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size,stride=s,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
    def forward(self,x):
        y = self.md(x)
        return self.bn1(self.conv2(y))+x