import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

embedding_dim = 256
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
        x1 = self.gn1(x1)
        x1 = F.gelu(x1)
        x1 = self.conv2(x1)
        x1 = self.gn2(x1)

        if self.use_residual:
            return F.gelu(x1 + x)
        else:
            return x1

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.out_channels = out_channels

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

        emb = self.projectEmbedding(t)
        emb = emb.view(-1, self.out_channels, 1, 1)

        return x + emb

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.out_channels = out_channels

        # used as opposite of max pool
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res1 = ConvResidualBlock(in_channels, in_channels, use_residual=True)
        self.res2 = ConvResidualBlock(in_channels, out_channels, use_residual=False)

        self.projectEmbedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels)
        )
    
    def forward(self, x, skip_x, raw_emb):
        x = self.upsample(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.res1(x)
        x = self.res2(x)

        emb = self.projectEmbedding(raw_emb)
        emb = emb.view(-1, self.out_channels, 1, 1)

        return x + emb

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ln1 = nn.LayerNorm([channels])
        self.ln2 = nn.LayerNorm([channels])
        self.linear1 = nn.Linear(channels, channels)
        self.linear2 = nn.Linear(channels, channels)
    
    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln1(x)
        x_mha, _ = self.mha(x_ln, x_ln, x_ln)
        x = x_mha + x
        x_tmp = self.ln2(x)
        x_tmp = F.gelu(self.linear1(x_tmp))
        x_tmp = self.linear2(x_tmp)
        x = x + x_tmp
        return x.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class UNet(nn.Module):
    def __init__(self, device, num_classes, img_size=32):
        super(UNet, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.img_size = img_size

        self.time_emb = self.initTimeEncoder()
        self.time_emb = self.time_emb.to(device)
        self.time_emb.requires_grad = False

        self.inp = ConvResidualBlock(in_channels=3, out_channels=64, use_residual=False)

        self.down1 = DownBlock(in_channels=64, out_channels=128)
        self.selfatt1 = SelfAttention(channels=128, size=self.img_size//2)
        self.down2 = DownBlock(in_channels=128, out_channels=256)
        self.selfatt2 = SelfAttention(channels=256, size=self.img_size//4)
        self.down3 = DownBlock(in_channels=256, out_channels=256)
        self.selfatt3 = SelfAttention(channels=256, size=self.img_size//8)

        self.middle1 = ConvResidualBlock(in_channels=256, out_channels=512, use_residual=False)
        self.middle2 = ConvResidualBlock(in_channels=512, out_channels=512, use_residual=False)
        self.middle3 = ConvResidualBlock(in_channels=512, out_channels=256, use_residual=False)

        self.up1 = UpBlock(in_channels=512, out_channels=128)
        self.selfatt4 = SelfAttention(channels=128, size=self.img_size//4)
        self.up2 = UpBlock(in_channels=256, out_channels=64)
        self.selfatt5 = SelfAttention(channels=64, size=self.img_size//2)
        self.up3 = UpBlock(in_channels=128, out_channels=64)
        self.selfatt6 = SelfAttention(channels=64, size=self.img_size)

        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1)

    def initTimeEncoder(self):
        posMat = torch.zeros((max_time, embedding_dim))
        for pos in range(max_time):
            for i in range(embedding_dim):
                if i % 2 == 0:
                    posMat[pos][i] = np.sin(pos / (10000 ** (i / embedding_dim)))
                else:
                    posMat[pos][i] = np.cos(pos / (10000 ** (i / embedding_dim)))
        return posMat

    def forward(self, x, t, labels=None):
        t = self.time_emb[t]

        if labels is not None:
            label_emb = self.label_embedding(labels)
            t += label_emb
        
        x1 = self.inp(x)
        x2 = self.down1(x1, t)
        x2 = self.selfatt1(x2)
        x3 = self.down2(x2, t)
        x3 = self.selfatt2(x3)
        x4 = self.down3(x3, t)
        x4 = self.selfatt3(x4)

        x = self.middle1(x4)
        x = self.middle2(x)
        x = self.middle3(x)

        x = self.up1(x, x3, t)
        x = self.selfatt4(x)
        x = self.up2(x, x2, t)
        x = self.selfatt5(x)
        x = self.up3(x, x1, t)
        x = self.selfatt6(x)

        x = self.out(x)
        return x
