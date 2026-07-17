import torch
import torch.nn as nn
from .ditac import DiTAC

class CBD3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CBD3DBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.ditac_mock = DiTAC(a=-3.0, b=3.0)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.ditac_mock(x)
        return x

class GCBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(GCBlock3D, self).__init__()
        self.context_modeling = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.transform = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1),
            nn.LayerNorm([in_channels // reduction, 1, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, T, H, W = x.size()
        context = self.context_modeling(x).view(B, 1, -1)
        context = torch.softmax(context, dim=-1)
        
        x_flat = x.view(B, C, -1)
        pool = torch.bmm(x_flat, context.transpose(1, 2)).unsqueeze(-1).unsqueeze(-1)
        
        transform = self.transform(pool)
        return x + transform

class FSFE(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super(FSFE, self).__init__()
        self.cbd1 = CBD3DBlock(in_channels, 16, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.cbd2 = CBD3DBlock(16, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gc_block = GCBlock3D(out_channels)

    def forward(self, x):
        x = self.cbd1(x)
        x = self.maxpool(x)
        x = self.cbd2(x)
        x = self.gc_block(x)
        return x