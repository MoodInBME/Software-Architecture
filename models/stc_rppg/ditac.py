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
        
        self.step = 1.0 / num_segments
        self.theta_raw = nn.Parameter(torch.zeros(num_segments + 1))
        
    def forward(self, x):
        out = x.clone()
        
        mask = (x >= self.a) & (x <= self.b)
        x_in = x[mask]
        
        if x_in.numel() > 0:
            phi_x = 0.5 * (1.0 + torch.erf(x_in / math.sqrt(2.0)))
            
            idx = torch.floor(phi_x / self.step).long()
            idx = torch.clamp(idx, 0, self.num_segments - 1)
            
            
            theta = torch.cumsum(torch.softmax(self.theta_raw, dim=0), dim=0)
            theta = theta / theta[-1].clamp_min(1e-8)   
            y1 = theta[idx + 1]
            x0 = idx * self.step
            
            y_interp = y0 + (phi_x - x0) * (y1 - y0) / self.step
            out[mask] = y_interp * (self.b - self.a) + self.a
            
        return out