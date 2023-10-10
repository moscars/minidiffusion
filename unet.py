import torch.nn as nn
import torch.nn.functional as F
import torch

class DownBlock(nn.Module):
    def __init__(self):
        super(DownBlock, self).__init__()
    
    def forward(self, x):
        pass

class UpBlock(nn.Module):
    def __init__(self):
        super(UpBlock, self).__init__()
    
    def forward(self, x):
        pass

class UnetEncoder(nn.Module):
    def __init__(self):
        super(UnetEncoder, self).__init__()
    
    def forward(self, x):
        pass

class UnetDecoder(nn.Module):
    def __init__(self):
        super(UnetDecoder, self).__init__()

        self.conv0 = nn.Conv2d(3, 64, 3, padding=1, bias=False)

    def forward(self, x):
        pass

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()