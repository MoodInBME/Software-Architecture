"""Konversi skala intensitas AU yang eksplisit.

Tiga sumber memakai skala berbeda:
  * label DISFA dan keluaran OpenFace ``AUxx_r``: 0..5;
  * kepala AU model (sigmoid) dan label di manifest: 0..1 (0..5 dibagi 5);
  * laporan metrik (ICC, MAE): dikembalikan ke 0..5 agar sebanding dengan pustaka.

Setiap perpindahan skala harus lewat fungsi di modul ini. Fungsi menolak nilai di luar
rentang, sehingga data 0..5 yang terlanjur dianggap 0..1 (atau sebaliknya) tertangkap
alih-alih diam-diam menjadi 0..25 atau 0..0,2. NaN dibiarkan (frame tanpa label).
"""

from __future__ import annotations

import numpy as np

AU_MAX = 5.0
_EPS = 1e-6


def _cek(x: np.ndarray, hi: float, nama: str, petunjuk: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    ok = a[~np.isnan(a)]
    if ok.size and (ok.min() < -_EPS or ok.max() > hi + _EPS):
        raise ValueError(
            f"{nama}: nilai di luar 0..{hi:g} (min {ok.min():.3f}, maks {ok.max():.3f}). {petunjuk}"
        )
    return a


def periksa_satuan(x, nama: str = "intensitas AU") -> np.ndarray:
    """Pastikan x di skala 0..1. Galat bila ada nilai > 1 (kemungkinan masih 0..5)."""
    return _cek(
        x, 1.0, nama, "Kemungkinan masih skala 0..5; ubah dulu dengan ke_satuan()."
    )


def ke_satuan(x05, nama: str = "intensitas AU") -> np.ndarray:
    """0..5 (DISFA, OpenFace AUxx_r) -> 0..1 (kepala AU)."""
    return _cek(x05, AU_MAX, nama, "Skala sumber harus 0..5.") / AU_MAX


def ke_skala_asli(x01, nama: str = "intensitas AU") -> np.ndarray:
    """0..1 (kepala AU) -> 0..5 (laporan metrik)."""
    return periksa_satuan(x01, nama) * AU_MAX
