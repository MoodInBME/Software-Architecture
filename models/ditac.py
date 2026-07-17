import torch
import torch.nn as nn
import math

class DiTAC(nn.Module):
    """
    DiTAC: Diffeomorphic Trajectory-based Activation Function (Approximation)
    Menggunakan transformasi Continuous Piecewise-Affine (CPA) yang dapat dilatih.
    """
    def __init__(self, num_segments=10, a=-3.0, b=3.0):
        super(DiTAC, self).__init__()
        self.a = a
        self.b = b
        self.num_segments = num_segments
        
        # Jarak interval (step) untuk segmentasi
        self.step = 1.0 / num_segments
        
        # Parameter yang bisa dilatih: Theta
        # Diinisialisasi secara linear agar pada awal pelatihan (epoch 0),
        # DiTAC bertindak mirip seperti fungsi aktivasi linear (identitas)
        self.theta = nn.Parameter(torch.linspace(0, 1, num_segments + 1))
        
    def forward(self, x):
        # 1. Di luar domain [a, b], fungsi bertindak sebagai identitas f(x) = x
        out = x.clone()
        
        # Masking untuk mencari nilai yang masuk ke dalam domain [a, b]
        mask = (x >= self.a) & (x <= self.b)
        x_in = x[mask]
        
        if x_in.numel() > 0:
            # 2. Menghitung CDF Normal Standar: Phi(x)
            # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
            phi_x = 0.5 * (1.0 + torch.erf(x_in / math.sqrt(2.0)))
            
            # 3. Transformasi Continuous Piecewise-Affine: T^theta(Phi(x))
            # Menentukan di segmen mana nilai Phi(x) jatuh (0 hingga num_segments - 1)
            idx = torch.floor(phi_x / self.step).long()
            
            # Keamanan batas indeks
            idx = torch.clamp(idx, 0, self.num_segments - 1)
            
            # Ambil nilai batas kiri (y0) dan kanan (y1) dari parameter Theta yang dipelajari
            y0 = self.theta[idx]
            y1 = self.theta[idx + 1]
            x0 = idx * self.step
            
            # Interpolasi affine (linear) pada segmen tersebut
            y_interp = y0 + (phi_x - x0) * (y1 - y0) / self.step
            
            # Petakan kembali output ke skala rentang spasial [a, b]
            out[mask] = y_interp * (self.b - self.a) + self.a
            
        return out