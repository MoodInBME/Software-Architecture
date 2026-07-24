import torch
import torch.nn as nn
import torch.nn.functional as F

class STCHybridLossFixed(nn.Module):
    def __init__(self, lambda_weight=0.5, fps=30.0, low=0.7, high=3.5):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.fps = fps
        self.low = low
        self.high = high

    def _band_logpsd(self, sig):
        n = sig.shape[1]
        fft = torch.fft.rfft(sig, dim=1)
        psd = fft.real ** 2 + fft.imag ** 2
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.fps).to(sig.device)
        band = (freqs >= self.low) & (freqs <= self.high)
        psd = psd[:, band]
        return F.log_softmax(psd + 1e-8, dim=1)

    def forward(self, predictions, targets):
        # Time-domain: Negative Pearson
        p = predictions - predictions.mean(1, keepdim=True)
        t = targets - targets.mean(1, keepdim=True)
        cov = (p * t).sum(1)
        denom = torch.sqrt((p ** 2).sum(1) + 1e-8) * torch.sqrt((t ** 2).sum(1) + 1e-8)
        loss_time = (1.0 - cov / denom).mean()

        # Frequency-domain: KL antara distribusi PSD (differentiable)
        logp = self._band_logpsd(predictions)
        logq = self._band_logpsd(targets)
        loss_freq = F.kl_div(logp, logq.exp(), reduction="batchmean")

        total = loss_time + self.lambda_weight * loss_freq
        return total, loss_time, loss_freq