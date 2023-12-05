import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        return F.relu(x + x1)
    
class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.device = device

        sz = 64
        self.conv1 = nn.Conv2d(3, sz, kernel_size=3)
        self.block1 = Block(sz)
        self.block2 = Block(sz)
        self.conv2 = nn.Conv2d(sz, sz, kernel_size=2, padding=1, stride=2)
        self.block3 = Block(sz)
        self.block4 = Block(sz)
        self.conv3 = nn.Conv2d(sz, 4, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        #print("Doing forward pass", x.shape)
        x = self.conv1(x)
        #print("After conv1", x.shape)
        x = F.relu(x)
        #print("After relu", x.shape)
        x = self.block1(x)
        #print("After block1", x.shape)
        x = self.block2(x)
        #print("After block2", x.shape)
        x = self.conv2(x)
        #print("After conv2", x.shape)
        x = F.relu(x)
        #print("After relu", x.shape)
        x = self.block3(x)
        #print("After block3", x.shape)
        x = self.block4(x)
        #print("After block4", x.shape)
        x = self.conv3(x)
        #print("After conv3", x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.block1 = Block(64)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.block2 = Block(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.block3 = Block(64)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.block4 = Block(64)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        #print("Doing forward pass", x.shape)
        x = self.conv1(x)
        #print("After conv1", x.shape)
        x = F.relu(x)
        #print("After relu", x.shape)
        x = self.block1(x)
        #print("After block1", x.shape)
        x = self.upsample1(x)
        #print("After upsample1", x.shape)
        x = self.block2(x)
        #print("After block2", x.shape)
        x = self.conv2(x)
        #print("After conv2", x.shape)
        x = F.relu(x)
        #print("After relu", x.shape)
        x = self.block3(x)
        #print("After block3", x.shape)
        x = self.upsample2(x)
        #print("After upsample2", x.shape)
        x = self.block4(x)
        #print("After block4", x.shape)
        x = self.conv3(x)
        #print("After conv3", x.shape)
        return x
    
class VariationalAutoEncoder(nn.Module):
    def __init__(self, device):
        super(VariationalAutoEncoder, self).__init__()
        self.device = device

        self.encoder = Encoder(device)
        self.decoder = Decoder(device)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)