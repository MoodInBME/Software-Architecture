import torch
import torch.nn as nn
import torch.nn.functional as F

class STCHybridLoss(nn.Module):
    def __init__(self, lambda_weight=0.5, fps=30.0, low=0.7, high=3.5):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.fps = fps
        self.low = low
        self.high = high

    def _band_logpsd(self, sig):
        # 1. Pastikan input selalu berpresisi Float32 untuk kestabilan FFT
        sig = sig.float()
        n = sig.shape[1]
        
        # 2. Hitung RFFT
        fft = torch.fft.rfft(sig, dim=1)
        psd = (fft.real ** 2 + fft.imag ** 2)
        
        # 3. Filter rentang frekuensi (misal: 0.7 Hz - 3.5 Hz atau 42 - 210 BPM)
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.fps).to(sig.device)
        band = (freqs >= self.low) & (freqs <= self.high)
        psd = psd[:, band]
        
        # 4. Log Softmax langsung pada PSD yang dipotong
        return F.log_softmax(psd, dim=1)

    def forward(self, predictions, targets):
        # Paksa tipe data Float32 agar Pearson & FFT tidak terkena underflow di AMP GPU T4
        predictions = predictions.float()
        targets = targets.float()

        # --- Time-domain: Negative Pearson Correlation ---
        p = predictions - predictions.mean(dim=1, keepdim=True)
        t = targets - targets.mean(dim=1, keepdim=True)
        
        cov = (p * t).sum(dim=1)
        denom = torch.sqrt((p ** 2).sum(dim=1) + 1e-8) * torch.sqrt((t ** 2).sum(dim=1) + 1e-8)
        
        # Hindari pembagian dengan nol
        pearson = cov / (denom + 1e-8)
        loss_time = (1.0 - pearson).mean()

        # --- Frequency-domain: KL Divergence pada PSD ---
        logp = self._band_logpsd(predictions)
        logq = self._band_logpsd(targets)
        
        # KL-Div mengharuskan input log_probabilities dan target probabilities
        loss_freq = F.kl_div(logp, logq.exp(), reduction="batchmean")

        # Total Loss
        total = loss_time + self.lambda_weight * loss_freq
        return total, loss_time, loss_freq
