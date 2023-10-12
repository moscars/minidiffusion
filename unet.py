import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

embedding_dim = 128
max_time = 1000

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
    def __init__(self, in_channels, out_channels, time_emb):
        super(DownBlock, self).__init__()
        self.time_emb = time_emb

        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.res1 = ConvResidualBlock(in_channels, in_channels, use_residual=True)
        self.res2 = ConvResidualBlock(in_channels, out_channels, use_residual=False)

        self.projectEmbedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels)
        )
    
    def forward(self, x, t):
        x = self.maxPool(x)
        x = self.res1(x)
        x = self.res2(x)

        raw_emb = self.time_emb[t]
        emb = self.projectEmbedding(raw_emb)

        return x + emb

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb):
        super(UpBlock, self).__init__()
        self.time_emb = time_emb

        # used as opposite of max pool
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res1 = ConvResidualBlock(in_channels, in_channels, use_residual=True)
        self.res2 = ConvResidualBlock(in_channels, out_channels, use_residual=False)

        self.projectEmbedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels)
        )
    
    def forward(self, x, skip_x, t):
        x = self.upsample(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.res1(x)
        x = self.res2(x)

        raw_emb = self.time_emb[t]
        emb = self.projectEmbedding(raw_emb)

        return x + emb

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
    
    def forward(self, x):
        pass

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.time_emb = self.initTimeEncoder()
        
        self.inp = ConvResidualBlock(in_channels=3, out_channels=64, use_residual=False)

        self.down1 = DownBlock(in_channels=64, out_channels=128, time_emb=self.time_emb)
        self.selfatt1 = SelfAttention()
        self.down2 = DownBlock(in_channels=128, out_channels=256, time_emb=self.time_emb)
        self.selfatt2 = SelfAttention()

        self.middle1 = ConvResidualBlock(in_channels=256, out_channels=512, use_residual=False)
        self.middle2 = ConvResidualBlock(in_channels=512, out_channels=256, use_residual=False)

        self.up1 = UpBlock()
        self.selfatt4 = SelfAttention()
        self.up2 = UpBlock()
        self.selfatt5 = SelfAttention()

        self.out = ConvResidualBlock

    def forward(self, x):
        pass

    def initTimeEncoder(self):
        posMat = torch.zeros((max_time, embedding_dim))
        for pos in range(max_time):
            for i in range(embedding_dim):
                if i % 2 == 0:
                    posMat[pos][i] = np.sin(pos / (10000 ** (i / embedding_dim)))
                else:
                    posMat[pos][i] = np.cos(pos / (10000 ** (i / embedding_dim)))
        return posMat