"""
Blok penyusun ST-GCN: graph convolution spasial + temporal convolution,
mengikuti Yan et al. (AAAI 2018).
"""
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """Spatial graph convolution (ConvTemporalGraphical)."""

    def __init__(self, in_channels, out_channels, k_num):
        super().__init__()
        self.k_num = k_num
        self.conv = nn.Conv2d(in_channels, out_channels * k_num, kernel_size=1)

    def forward(self, x, A):
        # x: (N, C, T, V) ; A: (K, V, V)
        x = self.conv(x)
        N, KC, T, V = x.size()
        x = x.view(N, self.k_num, KC // self.k_num, T, V)
        x = torch.einsum("nkctv,kvw->nctw", (x, A)).contiguous()
        return x


class STGCNBlock(nn.Module):
    """Graph conv spasial + temporal conv (Kt x 1) + koneksi residual."""

    def __init__(self, in_channels, out_channels, k_num, t_kernel=9,
                 stride=1, residual=True, dropout=0.0):
        super().__init__()
        pad = (t_kernel - 1) // 2
        self.gcn = GraphConvolution(in_channels, out_channels, k_num)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1),
                      (stride, 1), (pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)
