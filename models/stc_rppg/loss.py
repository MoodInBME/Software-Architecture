"""Rugi hibrida STC-rPPG: domain waktu dan domain frekuensi.

Mengikuti Lin dkk. (2025, IEEE TIM, Persamaan 8-10) dan rancangan Pra-TA
(Fiksasi Tahap B0, Persamaan 3.9-3.12):

    L_time = 1 - Pearson(s_pre, s_gt)                         (3.9)
    L_freq = |HR_pre - HR_gt|  (bpm)                          (3.10)
    HR_pre = 60 * sum_k f_k * softmax(P_k / tau)               (3.11)
    L      = L_time + lambda * L_freq,  lambda = 0,5           (3.12)

Paper tidak menjelaskan cara HR_pre dihitung saat pelatihan. Persamaan 3.11
adalah rancangan Pra-TA: soft-argmax spektral agar MAE HR dapat diturunkan.
P_k adalah daya spektrum ternormalisasi (dibagi puncaknya) pada bin pita HR,
setelah jendela Hann dan zero-padding ke n_fft titik. HR_gt dihitung dari
sinyal acuan dengan estimator yang sama tanpa gradien, sehingga prediksi yang
identik dengan acuan memberi L_freq = 0.

Pilihan freq_loss="kl_psd" mempertahankan rugi versi GEMASTIK (divergensi KL
antardistribusi PSD) untuk perbandingan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import HR_BAND_HZ

FREQ_LOSSES = ("mae_hr", "kl_psd")


class STCHybridLoss(nn.Module):
    def __init__(self, lambda_weight=0.5, fps=30.0, low=HR_BAND_HZ[0], high=HR_BAND_HZ[1],
                 freq_loss="mae_hr", tau=0.05, n_fft=1024):
        super().__init__()
        if freq_loss not in FREQ_LOSSES:
            raise ValueError(f"freq_loss harus salah satu dari {FREQ_LOSSES}, bukan {freq_loss!r}.")
        if tau <= 0:
            raise ValueError(f"tau harus positif, bukan {tau}.")
        self.lambda_weight = lambda_weight
        self.fps = fps
        self.low = low
        self.high = high
        self.freq_loss = freq_loss
        self.tau = float(tau)
        self.n_fft = int(n_fft)

    # ------------------------------------------------------------------
    # Domain frekuensi, pilihan "mae_hr" (bawaan)
    # ------------------------------------------------------------------
    def _band_spectrum(self, sig):
        """Daya spektrum pada pita HR setelah rerata dibuang, jendela Hann, dan zero-padding."""
        sig = sig.float()
        n = sig.shape[1]
        n_fft = max(self.n_fft, n)
        sig = sig - sig.mean(dim=1, keepdim=True)
        sig = sig * torch.hann_window(n, periodic=False, dtype=sig.dtype, device=sig.device)
        spec = torch.fft.rfft(sig, n=n_fft, dim=1)
        power = spec.real ** 2 + spec.imag ** 2
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / self.fps).to(sig.device)
        band = (freqs >= self.low) & (freqs <= self.high)
        if not bool(band.any()):
            raise ValueError(f"Tidak ada bin FFT pada pita {self.low}-{self.high} Hz.")
        return freqs[band], power[:, band]

    def soft_hr(self, sig):
        """HR terdiferensialkan (bpm) dengan soft-argmax spektral, Persamaan 3.11."""
        freqs, power = self._band_spectrum(sig)
        rel = power / power.amax(dim=1, keepdim=True).clamp_min(1e-12)
        weights = torch.softmax(rel / self.tau, dim=1)
        return 60.0 * (weights * freqs).sum(dim=1)

    # ------------------------------------------------------------------
    # Domain frekuensi, pilihan "kl_psd" (versi GEMASTIK, untuk perbandingan)
    # ------------------------------------------------------------------
    def _band_logpsd(self, sig):
        # 1. Pastikan input selalu berpresisi Float32 untuk kestabilan FFT
        sig = sig.float()
        n = sig.shape[1]

        # 2. Hitung RFFT
        fft = torch.fft.rfft(sig, dim=1)
        psd = (fft.real ** 2 + fft.imag ** 2)

        # 3. Batasi ke pita HR (bawaan constants.HR_BAND_HZ: 0,75-4,0 Hz atau 45-240 BPM)
        freqs = torch.fft.rfftfreq(n, d=1.0 / self.fps).to(sig.device)
        band = (freqs >= self.low) & (freqs <= self.high)
        psd = psd[:, band]

        # 4. Log Softmax langsung pada PSD yang dipotong
        return F.log_softmax(psd, dim=1)

    def forward(self, predictions, targets):
        # Paksa tipe data Float32 agar Pearson & FFT tidak terkena underflow di AMP GPU T4
        predictions = predictions.float()
        targets = targets.float()

        # --- Domain waktu: Negative Pearson, Persamaan 3.9 ---
        p = predictions - predictions.mean(dim=1, keepdim=True)
        t = targets - targets.mean(dim=1, keepdim=True)

        cov = (p * t).sum(dim=1)
        denom = torch.sqrt((p ** 2).sum(dim=1) + 1e-8) * torch.sqrt((t ** 2).sum(dim=1) + 1e-8)

        # Hindari pembagian dengan nol
        pearson = cov / (denom + 1e-8)
        loss_time = (1.0 - pearson).mean()

        # --- Domain frekuensi ---
        if self.freq_loss == "mae_hr":
            # Persamaan 3.10-3.11; satuan bpm
            with torch.no_grad():
                hr_gt = self.soft_hr(targets)
            loss_freq = (self.soft_hr(predictions) - hr_gt).abs().mean()
        else:
            # KL-Div mengharuskan input log_probabilities dan target probabilities
            logp = self._band_logpsd(predictions)
            logq = self._band_logpsd(targets)
            loss_freq = F.kl_div(logp, logq.exp(), reduction="batchmean")

        # Total, Persamaan 3.12
        total = loss_time + self.lambda_weight * loss_freq
        return total, loss_time, loss_freq
