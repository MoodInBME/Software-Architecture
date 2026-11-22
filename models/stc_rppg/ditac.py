"""DiTAC: fungsi aktivasi terlatih berbasis transformasi difeomorfik (Lin dkk., 2025).

Persamaan (1) paper rujukan STC-rPPG:

    DiTAC(x) = x_tilde * Phi(x),
    x_tilde  = a + (b - a) * T((x - a) / (b - a))   bila a <= x <= b,
    x_tilde  = x                                     di luar [a, b],

dengan Phi fungsi distribusi kumulatif normal baku dan T transformasi monoton
dari [0, 1] ke [0, 1] yang dapat dilatih. Domain bawaan [a, b] = [-3, 3].

Penyimpangan dari paper (disengaja, dicatat juga di README):
  * Paper memakai T berupa transformasi CPAB (integrasi medan kecepatan affine
    sepotong-sepotong). Di sini T adalah interpolasi linear sepotong-sepotong
    antara ``num_segments + 1`` simpul yang naik monoton. Keduanya monoton dan
    memetakan [0, 1] ke [0, 1], tetapi T versi ini hanya kontinu, tidak mulus di
    simpul, sehingga turunannya melompat di batas segmen.
  * Paper menulis DiTAC tidak menambah parameter; versi ini punya
    ``num_segments`` parameter per modul (``theta_raw``).

Inisialisasi ``theta_raw = 0`` memberi simpul berjarak sama, yaitu T identitas,
sehingga pada awal pelatihan DiTAC(x) = x * Phi(x) = GELU (bentuk erf) di semua x.
Karena T(0) = 0 dan T(1) = 1, keluaran kontinu di x = a dan x = b.
"""

import math

import torch
import torch.nn as nn


class DiTAC(nn.Module):
    def __init__(self, num_segments=10, a=-3.0, b=3.0):
        super().__init__()
        if not b > a:
            raise ValueError(f"Domain harus a < b, dapat a={a}, b={b}")
        if num_segments < 1:
            raise ValueError("num_segments harus >= 1")
        self.a = float(a)
        self.b = float(b)
        self.num_segments = int(num_segments)
        # Kenaikan tiap segmen lewat softmax: selalu positif dan berjumlah 1,
        # sehingga simpul hasil cumsum naik monoton dari 0 ke 1.
        self.theta_raw = nn.Parameter(torch.zeros(self.num_segments))

    def knots(self) -> torch.Tensor:
        """Nilai T di simpul u_k = k / num_segments, k = 0..num_segments."""
        inc = torch.softmax(self.theta_raw, dim=0)
        zero = torch.zeros(1, device=inc.device, dtype=inc.dtype)
        return torch.cat([zero, torch.cumsum(inc, dim=0)])

    def transform_unit(self, u: torch.Tensor) -> torch.Tensor:
        """T(u) untuk u di [0, 1] (nilai di luar dipotong ke [0, 1])."""
        u = u.clamp(0.0, 1.0)
        theta = self.knots().to(u.dtype)
        s = u * self.num_segments
        idx = torch.clamp(torch.floor(s).long(), 0, self.num_segments - 1)
        frac = s - idx.to(u.dtype)
        y0 = theta[idx]
        y1 = theta[idx + 1]
        return y0 + frac * (y1 - y0)

    def forward(self, x):
        # Dihitung untuk semua elemen lalu dipilih dengan torch.where: tanpa
        # pengindeksan boolean, sehingga tidak ada tensor berukuran variabel.
        phi = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        width = self.b - self.a
        u = (x - self.a) / width
        inside = (x >= self.a) & (x <= self.b)
        x_tilde = torch.where(inside, self.a + width * self.transform_unit(u), x)
        return x_tilde * phi
