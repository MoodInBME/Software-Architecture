import torch
import torch.nn as nn
from .ditac import DiTAC

class SpatiotemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatiotemporalConvBlock, self).__init__()
        self.cdc_st_mock = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(out_channels)
        self.ditac_mock = DiTAC(a=-3.0, b=3.0)

    def forward(self, x):
        x = self.cdc_st_mock(x)
        x = self.bn(x)
        x = self.ditac_mock(x)
        return x

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.stc1 = SpatiotemporalConvBlock(in_channels, out_channels)
        self.stc2 = SpatiotemporalConvBlock(out_channels, out_channels)
        self.maxpool_main = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        self.maxpool_shortcut = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.shortcut_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        main = self.maxpool_main(self.stc2(self.stc1(x)))
        shortcut = self.shortcut_conv(self.maxpool_shortcut(x))
        return main + shortcut

class STFE(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(STFE, self).__init__()
        self.res_block1 = ResidualBlock3D(in_channels, out_channels)
        self.res_block2 = ResidualBlock3D(out_channels, out_channels)
        self.final_conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.final_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x