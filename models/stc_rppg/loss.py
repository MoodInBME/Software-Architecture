import torch
import torch.nn as nn
import torch.nn.functional as F

class STCHybridLoss(nn.Module):
    """
    Time- and Frequency-Domain Hybrid Loss untuk estimasi rPPG
    Total Loss = L_time + lambda * L_freq
    """
    def __init__(self, lambda_weight=0.5, fps=30.0):
        super(STCHybridLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.fps = fps

    def forward(self, predictions, targets):
        """
        predictions: Sinyal rPPG prediksi dari model, shape: (Batch, Time)
        targets: Sinyal rPPG ground-truth (BVP), shape: (Batch, Time)
        """
        # ==========================================
        # 1. TIME-DOMAIN LOSS (Negative Pearson)
        # ==========================================
        # Normalisasi ke zero-mean
        pred_mean = torch.mean(predictions, dim=1, keepdim=True)
        target_mean = torch.mean(targets, dim=1, keepdim=True)
        
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        
        # Hitung Covariance dan Standard Deviation
        cov = torch.sum(pred_centered * target_centered, dim=1)
        pred_std = torch.sqrt(torch.sum(pred_centered**2, dim=1) + 1e-8)
        target_std = torch.sqrt(torch.sum(target_centered**2, dim=1) + 1e-8)
        
        # Pearson Correlation Coefficient (r)
        pearson = cov / (pred_std * target_std)
        
        # L_time = 1 - r
        loss_time = 1.0 - pearson
        loss_time = torch.mean(loss_time)

        # ==========================================
        # 2. FREQUENCY-DOMAIN LOSS (MAE dari Puncak FFT)
        # ==========================================
        # Menghitung Power Spectral Density (PSD) menggunakan Fast Fourier Transform (FFT)
        # Menggunakan torch.fft.rfft untuk sinyal real 1D
        pred_fft = torch.fft.rfft(predictions, dim=1)
        target_fft = torch.fft.rfft(targets, dim=1)
        
        # Mengambil magnitudo spektrum (Power)
        pred_psd = torch.abs(pred_fft)
        target_psd = torch.abs(target_fft)
        
        # Mengestimasi frekuensi dominan (Detak Jantung)
        # Indeks dengan nilai energi terbesar diasumsikan sebagai frekuensi HR
        pred_hr_idx = torch.argmax(pred_psd, dim=1).float()
        target_hr_idx = torch.argmax(target_psd, dim=1).float()
        
        # Menghitung MAE antara indeks frekuensi prediksi dan target
        loss_freq = F.l1_loss(pred_hr_idx, target_hr_idx)
        
        # ==========================================
        # TOTAL LOSS
        # ==========================================
        total_loss = loss_time + (self.lambda_weight * loss_freq)
        
        return total_loss, loss_time, loss_freq