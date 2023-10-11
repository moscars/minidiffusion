import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual):
        super(ConvResidualBlock, self).__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(1, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(1, out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.gn1(x1)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.gn2(x)

        if self.use_residual:
            x = F.gelu(x1 + x)
        
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()

        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.res1 = ConvResidualBlock(in_channels, in_channels, use_residual=True)
        self.res2 = ConvResidualBlock(in_channels, out_channels, use_residual=False)
    
    def forward(self, x, t):
        x = self.maxPool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        # used as opposite of max pool
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res1 = ConvResidualBlock(in_channels, in_channels, use_residual=True)
        self.res2 = ConvResidualBlock(in_channels, out_channels, use_residual=False)
    
    def forward(self, x, skip_x):
        x = self.upsample(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.res1(x)
        x = self.res2(x)

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
    
    def forward(self, x):
        pass

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.inp = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.down1 = DownBlock()
        self.selfatt1 = SelfAttention()
        self.down2 = DownBlock()
        self.selfatt2 = SelfAttention()

        self.middle1 = None
        self.middle2 = None

        self.up1 = UpBlock()
        self.selfatt4 = SelfAttention()
        self.up2 = UpBlock()
        self.selfatt5 = SelfAttention()

        self.out = nn.Conv2d()

    def forward(self, x):
        pass