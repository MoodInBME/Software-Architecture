import torch
import torch.nn as nn
import torch.nn.functional as F


class CDC_ST(nn.Conv3d):
    """3D Spatiotemporal Central Difference Convolution.

    Mengikuti formula central difference convolution (Yu et al., referensi
    [27] pada paper STC-rPPG): output = konvolusi normal - theta * (nilai
    piksel pusat dikonvolusi dengan jumlah bobot kernel). Term kedua ini
    menangkap perbedaan lokal (gradien) antar piksel tetangga terhadap pusat,
    yang secara empiris lebih baik menekan noise/redundansi dibanding
    konvolusi biasa untuk sinyal rPPG (lihat ablasi Table VII di paper:
    3D-CDC-ST > 3DConv, R(2+1)D, 3D-CDC-T).

    theta=0 -> setara persis dengan nn.Conv3d biasa (berguna untuk sanity
    check / ablasi).
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3),
                 stride=1, padding=(1, 1, 1), dilation=1, groups=1,
                 bias=True, theta=0.6):
        super().__init__(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        self.theta = theta

    def forward(self, x):
        out_normal = F.conv3d(
            x, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )

        if abs(self.theta) < 1e-8:
            return out_normal

        # Jumlahkan bobot kernel di seluruh dimensi spasial-temporal (kT,kH,kW)
        # -> jadi kernel 1x1x1 per pasangan (out_channel, in_channel/groups).
        # Konvolusi dengan kernel 1x1x1 padding 0 setara mengambil nilai
        # piksel pusat dan mengalikannya dengan total bobot kernel asli.
        kernel_diff = self.weight.sum(dim=(2, 3, 4), keepdim=True)
        out_diff = F.conv3d(
            x, kernel_diff, None, self.stride,
            padding=0, dilation=self.dilation, groups=self.groups,
        )

        return out_normal - self.theta * out_diff


if __name__ == "__main__":
    # Sanity check cepat: theta=0 harus identik dengan Conv3d biasa,
    # dan output shape harus konsisten dengan padding "same".
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 32, 32)

    cdc0 = CDC_ST(8, 16, theta=0.0)
    plain = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
    plain.weight.data = cdc0.weight.data.clone()
    plain.bias.data = cdc0.bias.data.clone()

    out_cdc0 = cdc0(x)
    out_plain = plain(x)
    print("theta=0 identik dengan Conv3d biasa:",
          torch.allclose(out_cdc0, out_plain, atol=1e-6))

    cdc = CDC_ST(8, 16, theta=0.6)
    out = cdc(x)
    print("Output shape (theta=0.6):", tuple(out.shape))
    assert out.shape == out_plain.shape, "Shape CDC-ST harus sama dgn Conv3d biasa!"
    print("Shape konsisten: OK")