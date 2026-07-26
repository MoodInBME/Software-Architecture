import torch
import torch.nn as nn
import math

class DiTAC(nn.Module):
    """
    DiTAC: Diffeomorphic Trajectory-based Activation Function (Approximation)
    Menggunakan transformasi Continuous Piecewise-Affine (CPA) yang dapat dilatih.

    FIX (vs versi awal): theta_raw sekarang berukuran `num_segments` (bukan
    num_segments+1) dan dipakai sebagai increment antar-segmen lewat softmax,
    lalu di-cumsum dengan 0 di depan. Ini menjamin theta[0]=0 dan theta[-1]=1,
    sehingga transformasi benar-benar memetakan [0,1] -> [0,1] secara monoton,
    dan titik ujung domain (x=a, x=b) tidak lagi bergeser dari nilai aslinya.
    """
    def __init__(self, num_segments=10, a=-3.0, b=3.0):
        super(DiTAC, self).__init__()
        self.a = a
        self.b = b
        self.num_segments = num_segments

        self.step = 1.0 / num_segments
        # increment tiap segmen (bukan titik theta langsung) -> dijamin positif
        # lewat softmax, sehingga theta hasil cumsum otomatis monoton naik.
        self.theta_raw = nn.Parameter(torch.zeros(num_segments))

    def forward(self, x):
        # FIX (memory): versi sebelumnya pakai boolean-mask indexing
        # (x[mask], out[mask]=...) yang bikin PyTorch alokasi tensor baru
        # + buffer scatter untuk backward -> boros VRAM, terbukti OOM di
        # GPU 14-15GB (Colab T4) saat batch besar. Versi ini menghitung
        # transformasi untuk SEMUA elemen sekaligus (vectorized, elementwise)
        # lalu pilih hasilnya lewat torch.where -> tidak ada tensor
        # berukuran variabel, tidak ada scatter, jauh lebih hemat memori.
        phi_x = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

        idx = torch.floor(phi_x / self.step).long()
        idx = torch.clamp(idx, 0, self.num_segments - 1)

        increments = torch.softmax(self.theta_raw, dim=0)  # jumlah = 1
        zero = torch.zeros(1, device=x.device, dtype=increments.dtype)
        theta = torch.cat([zero, torch.cumsum(increments, dim=0)])  # theta[0]=0, theta[-1]=1

        y0 = theta[idx]
        y1 = theta[idx + 1]
        x0 = idx.to(phi_x.dtype) * self.step

        y_interp = y0 + (phi_x - x0) * (y1 - y0) / self.step
        transformed = y_interp * (self.b - self.a) + self.a

        mask = (x >= self.a) & (x <= self.b)
        return torch.where(mask, transformed, x)