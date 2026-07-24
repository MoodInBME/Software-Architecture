import torch
import torch.nn as nn
from .ditac import DiTAC
from .cdc import CDC_ST

class SpatiotemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cdc_theta=0.6):
        super(SpatiotemporalConvBlock, self).__init__()
        self.cdc_st = CDC_ST(in_channels, out_channels, kernel_size=(3, 3, 3),
                              padding=(1, 1, 1), theta=cdc_theta)
        self.bn = nn.BatchNorm3d(out_channels)
        self.ditac = DiTAC(a=-3.0, b=3.0)

    def forward(self, x):
        x = self.cdc_st(x)
        x = self.bn(x)
        x = self.ditac(x)
        return x

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, cdc_theta=0.6):
        super(ResidualBlock3D, self).__init__()
        self.stc1 = SpatiotemporalConvBlock(in_channels, out_channels, cdc_theta=cdc_theta)
        self.stc2 = SpatiotemporalConvBlock(out_channels, out_channels, cdc_theta=cdc_theta)
        self.maxpool_main = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.maxpool_shortcut = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        # Eq. (6) paper: shortcut juga pakai CDC_ST, bukan Conv3d biasa.
        self.shortcut_cdc = CDC_ST(in_channels, out_channels, kernel_size=(3, 3, 3),
                                    padding=(1, 1, 1), theta=cdc_theta)

    def forward(self, x):
        main = self.maxpool_main(self.stc2(self.stc1(x)))
        shortcut = self.shortcut_cdc(self.maxpool_shortcut(x))
        return main + shortcut

class STFE(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, cdc_theta=0.6):
        super(STFE, self).__init__()
        self.res_block1 = ResidualBlock3D(in_channels, out_channels, cdc_theta=cdc_theta)
        self.res_block2 = ResidualBlock3D(out_channels, out_channels, cdc_theta=cdc_theta)
        self.final_conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.final_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x