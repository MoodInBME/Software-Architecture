# Tinjauan Kode STC-rPPG (Review Kritis + Patch)

Dokumen ini meninjau file komponen STC-rPPG yang sudah ada dan memberi patch
siap-tempel. **File asli tidak diubah** agar hasil kerja tim tetap utuh;
terapkan patch bila setuju.

Ringkasan tingkat keparahan:

| # | Temuan | Keparahan | Dampak |
|---|---|---|---|
| 1 | Loss frekuensi memakai `argmax` (non-differentiable) | **Tinggi** | Komponen loss frekuensi tidak menghasilkan gradien -> tidak ikut melatih model |
| 2 | Blok `cdc_st_mock` & `ditac_mock` bernama "mock" | **Tinggi (integritas)** | Klaim CDC/DiTAC di proposal belum benar-benar terimplementasi |
| 3 | `DiTAC.theta` tidak dijamin monoton | Sedang | Transformasi belum tentu diffeomorphic (kontradiksi definisi DiTAC) |
| 4 | Python 3.14 | **Tinggi** | Wheel PyTorch bisa belum tersedia -> lingkungan gagal dibangun |
| 5 | Skala `loss_freq` vs `loss_time` | Rendah | Perlu penyetelan `lambda_weight` |

---

## 1. [Tinggi] Loss frekuensi non-differentiable

Di `loss.py`, frekuensi dominan diambil via `torch.argmax(pred_psd)` lalu
dihitung L1 terhadap `argmax(target_psd)`. **`argmax` tidak punya gradien**,
sehingga `loss_freq` **tidak berkontribusi apa pun** ke pembaruan bobot.
Efektifnya model hanya dilatih oleh Negative-Pearson (time-domain).

**Perbaikan:** bandingkan distribusi PSD (bukan indeks argmax) di pita HR
fisiologis (0.7-3.5 Hz ~ 42-210 bpm) memakai KL-divergence yang differentiable.

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

---

## 2. [Tinggi/integritas] Blok "mock" belum implementasi CDC

Di `stfe.py` (`cdc_st_mock`) dan `fsfe.py` (`ditac_mock`) memakai `nn.Conv3d`
biasa dan penamaan `_mock`. Jika proposal mengklaim **Central Difference
Convolution (CDC)**, klaim itu belum terpenuhi. Ganti dengan CDC sungguhan
(atau ubah klaim di proposal agar jujur).

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CDC3D(nn.Module):
        """Central Difference Convolution 3D. theta menyeimbangkan konv biasa
        vs suku central-difference (theta=0 -> Conv3d biasa)."""
        def __init__(self, in_c, out_c, kernel_size=3, padding=1, theta=0.6):
            super().__init__()
            self.conv = nn.Conv3d(in_c, out_c, kernel_size, padding=padding, bias=False)
            self.theta = theta

        def forward(self, x):
            out_normal = self.conv(x)
            if abs(self.theta) < 1e-6:
                return out_normal
            kernel_diff = self.conv.weight.sum(dim=(2, 3, 4), keepdim=True)
            out_diff = F.conv3d(x, kernel_diff, stride=self.conv.stride, padding=0)
            return out_normal - self.theta * out_diff

Ganti `self.cdc_st_mock = nn.Conv3d(...)` menjadi `self.cdc_st = CDC3D(...)`.

---

## 3. [Sedang] DiTAC belum monoton (diffeomorphic)

Di `ditac.py`, `self.theta = nn.Parameter(torch.linspace(0, 1, N+1))` bebas
berubah saat training sehingga bisa **tidak monoton naik** -> transformasi tak
lagi diffeomorphic (melanggar definisi DiTAC). Paksa monoton via cumulative
softmax.

    # __init__:
    self.theta_raw = nn.Parameter(torch.zeros(num_segments + 1))

    # forward (ganti pemakaian self.theta):
    theta = torch.cumsum(torch.softmax(self.theta_raw, dim=0), dim=0)
    theta = theta / theta[-1].clamp_min(1e-8)   # ternormalisasi ke [0, 1], monoton naik
    y0 = theta[idx]
    y1 = theta[idx + 1]

---

## 4. [Tinggi] Python 3.14

PyTorch umumnya tertinggal 1-2 versi minor Python. Pada 3.14 wheel resmi bisa
belum tersedia sehingga `pip install torch` gagal / harus build dari sumber.

**Rekomendasi:** kunci ke **Python 3.11 atau 3.12** untuk repo ini. Untuk
target deployment Jetson, JetPack 6.2 memakai Python 3.10/3.12 (bukan 3.14).
Menyamakan versi sekarang mencegah kejutan saat porting ke Jetson/TensorRT.

---

## 5. [Rendah] Skala loss

`loss_time` (Negative Pearson) berada di rentang ~[0, 2], sedangkan `loss_freq`
lama (MAE indeks bin) bisa berskala puluhan. Setelah patch #1, keduanya
berskala serupa; `lambda_weight` awal 0.5 wajar, sesuaikan lewat validasi.

---

## Catatan verifikasi

Semua file STC-rPPG **lulus `py_compile`** (sintaks valid). Eksekusi numerik
belum dijalankan karena PyTorch tidak terpasang di lingkungan build (jaringan
mati). Uji `forward`/`backward` sebaiknya dijalankan di lingkungan tim dengan
PyTorch terpasang.
