import torch
import torch.nn as nn

class rPPGPredictor(nn.Module):
    def __init__(self, in_channels=32):
        super(rPPGPredictor, self).__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(in_channels),
            nn.ELU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(in_channels),
            nn.ELU(inplace=True)
        )
        self.adap_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.final_conv = nn.Conv3d(in_channels, 1, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.adap_pool(x)
        x = self.final_conv(x)
        return x.squeeze(-1).squeeze(-1).squeeze(1)