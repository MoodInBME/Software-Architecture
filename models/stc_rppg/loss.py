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
        # 1. Paksa operasi FFT keluar dari AMP ke Float32 untuk menghindari bug ComplexHalf
        with torch.cuda.amp.autocast(enabled=False):
            sig_f32 = sig.float()
            n = sig_f32.shape[1]
            fft = torch.fft.rfft(sig_f32, dim=1)
            psd = fft.real ** 2 + fft.imag ** 2
            
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.fps).to(sig.device)
        band = (freqs >= self.low) & (freqs <= self.high)
        psd = psd[:, band]
        
        # Kembalikan ke tipe data asli sig agar sinkron dengan kalkulasi setelahnya
        psd = psd.to(sig.dtype)
        return F.log_softmax(psd + 1e-8, dim=1)

    def forward(self, predictions, targets):
        # Time-domain: Negative Pearson
        p = predictions - predictions.mean(1, keepdim=True)
        t = targets - targets.mean(1, keepdim=True)
        cov = (p * t).sum(1)
        
        # 2. Pindahkan penambahan epsilon (+ 1e-8) ke dalam torch.sqrt untuk mencegah akar dari nol (sqrt(0) -> gradien nan)
        denom = torch.sqrt((p ** 2).sum(1) + 1e-8) * torch.sqrt((t ** 2).sum(1) + 1e-8)
        
        # 3. Berikan pengaman tambahan pada pembagian jika denom sangat kecil
        loss_time = (1.0 - cov / (denom + 1e-8)).mean()

        # Frequency-domain: KL antara distribusi PSD (differentiable)
        logp = self._band_logpsd(predictions)
        logq = self._band_logpsd(targets)
        
        # 4. Gunakan stabilitas numerik untuk logq.exp() agar nilainya tidak meledak atau bernilai negatif karena underflow AMP
        q_prob = torch.clamp(logq.exp(), min=1e-8, max=1.0)
        loss_freq = F.kl_div(logp, q_prob, reduction="batchmean")

        total = loss_time + self.lambda_weight * loss_freq
        return total, loss_time, loss_freq
