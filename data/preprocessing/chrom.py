"""CHROM (de Haan dan Jeanne, IEEE TBME 2013): rPPG berbasis krominans sebagai pembanding klasik.

Masukan: deret rerata RGB kulit per frame (T, 3). Per jendela pendek (bawaan 1,6 s,
geser setengah jendela) warna dinormalisasi dengan reratanya, dibentuk
X = 3R - 2G dan Y = 1,5R + G - 1,5B, keduanya difilter pita 0,75-4,0 Hz,
S = X - (std X / std Y) Y, lalu S diberi jendela Hann dan dijumlahkan bertumpuk.

ROI kulit diambil dari landmark FaceMesh (dahi dan dua pipi); hanya rerata RGB yang
disimpan, frame dibuang setelah dihitung.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from models.stc_rppg.constants import HR_BAND_HZ

# Indeks FaceMesh untuk poligon dahi dan dua pipi [Sedang: indeks yang umum dipakai
# implementasi rPPG berbasis FaceMesh; periksa visual sebelum dipakai di data primer].
POLIGON_ROI = {
    "dahi": [67, 109, 10, 338, 297, 299, 9, 69],
    "pipi_kanan": [117, 118, 101, 36, 205, 187, 123],
    "pipi_kiri": [346, 347, 330, 266, 425, 411, 352],
}


def chrom(
    rgb: np.ndarray, fs: float, win_s: float = 1.6, band=HR_BAND_HZ, order: int = 3
) -> np.ndarray:
    """Sinyal pulsa (T,) dari rerata RGB (T, 3). Frame dengan NaN diisi interpolasi linear."""
    c = np.asarray(rgb, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"rgb harus (T, 3), dapat {c.shape}")
    if fs <= 2 * band[1]:
        raise ValueError(f"fs {fs} terlalu rendah untuk pita {band}")
    T = c.shape[0]
    ok = np.isfinite(c).all(axis=1)
    if ok.sum() < 2:
        raise ValueError("Kurang dari dua frame dengan ROI valid")
    if not ok.all():
        t = np.arange(T)
        c = np.stack([np.interp(t, t[ok], c[ok, k]) for k in range(3)], axis=1)
    L = int(round(win_s * fs))
    L += L % 2
    if T < L:
        raise ValueError(f"Sinyal {T} frame lebih pendek dari jendela {L}")
    b, a = butter(order, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    hann = np.hanning(L)
    out = np.zeros(T)
    for s in range(0, T - L + 1, L // 2):
        w = c[s : s + L]
        mu = w.mean(axis=0)
        if np.any(mu <= 0):
            continue
        n = w / mu
        x = 3 * n[:, 0] - 2 * n[:, 1]
        y = 1.5 * n[:, 0] + n[:, 1] - 1.5 * n[:, 2]
        xf, yf = filtfilt(b, a, x), filtfilt(b, a, y)
        sy = yf.std()
        alpha = xf.std() / sy if sy > 0 else 0.0
        seg = xf - alpha * yf
        out[s : s + L] += hann * (seg - seg.mean())
    return out


def rerata_rgb_roi(bgr: np.ndarray, titik_px: np.ndarray) -> np.ndarray:
    """Rerata (R, G, B) di dalam gabungan poligon ROI. titik_px: (468, 2) piksel."""
    import cv2

    mask = np.zeros(bgr.shape[:2], np.uint8)
    for idx in POLIGON_ROI.values():
        cv2.fillPoly(mask, [np.round(titik_px[idx]).astype(np.int32)], 1)
    if mask.sum() == 0:
        return np.full(3, np.nan)
    b, g, r, _ = cv2.mean(bgr, mask=mask)
    return np.array([r, g, b])
