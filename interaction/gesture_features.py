"""Fitur tangan ternormalisasi dari 21 titik sendi MediaPipe Hands.

Indeks titik (konvensi MediaPipe Hands): 0 pergelangan; ibu jari 1-4; telunjuk 5-8;
jari tengah 9-12; jari manis 13-16; kelingking 17-20 (urutan MCP, PIP, DIP, ujung;
ibu jari CMC, MCP, IP, ujung).

Semua fitur tak bergantung pada posisi dan ukuran tangan di gambar: titik digeser
relatif ke pergelangan lalu dibagi ukuran telapak (jarak pergelangan ke pangkal jari
tengah, titik 9). Titik sendi hanya dipakai di memori; tidak disimpan dan tidak dikirim.
"""

from __future__ import annotations

import numpy as np

N_TITIK = 21
PERGELANGAN = 0
PANGKAL_TELUNJUK = 5  # posisi kursor (stabil saat mencubit)
PANGKAL_TENGAH = 9
UJUNG_IBU_JARI = 4
UJUNG_TELUNJUK = 8

# (ujung, PIP) untuk telunjuk, tengah, manis, kelingking.
_JARI = ((8, 6), (12, 10), (16, 14), (20, 18))


def _cek(lm) -> np.ndarray:
    a = np.asarray(lm, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != N_TITIK or a.shape[1] < 2:
        raise ValueError(f"Titik sendi harus ({N_TITIK}, >=2), dapat {a.shape}")
    a = a[:, :2]
    if not np.isfinite(a).all():
        raise ValueError("Titik sendi memuat NaN atau inf")
    return a


def ukuran_telapak(lm) -> float:
    a = _cek(lm)
    return float(np.linalg.norm(a[PANGKAL_TENGAH] - a[PERGELANGAN]))


def normalisasi_tangan(lm, eps: float = 1e-6) -> np.ndarray:
    """(21, 2) relatif ke pergelangan dan dibagi ukuran telapak."""
    a = _cek(lm)
    s = np.linalg.norm(a[PANGKAL_TENGAH] - a[PERGELANGAN])
    if s < eps:
        raise ValueError("Ukuran telapak nol; titik sendi tidak valid")
    return (a - a[PERGELANGAN]) / s


def rasio_cubit(lm) -> float:
    """Jarak ujung ibu jari ke ujung telunjuk dibagi ukuran telapak."""
    n = normalisasi_tangan(lm)
    return float(np.linalg.norm(n[UJUNG_IBU_JARI] - n[UJUNG_TELUNJUK]))


def jari_lurus(lm, faktor: float = 1.15) -> np.ndarray:
    """Bool (4,) untuk telunjuk, tengah, manis, kelingking.

    Jari dianggap lurus bila ujungnya lebih jauh dari pergelangan daripada sendi
    PIP-nya, dikali ``faktor``.
    """
    n = normalisasi_tangan(lm)
    jarak = np.linalg.norm(n, axis=1)
    return np.array([jarak[u] > faktor * jarak[p] for u, p in _JARI])


def ibu_jari_terbuka(lm, ambang: float = 0.7) -> bool:
    """Ibu jari menjauh dari pangkal telunjuk (jarak ternormalisasi titik 4 ke 5)."""
    n = normalisasi_tangan(lm)
    return bool(np.linalg.norm(n[UJUNG_IBU_JARI] - n[PANGKAL_TELUNJUK]) > ambang)


def telapak_terbuka(lm) -> bool:
    """Lima jari terbuka (gestur kembali bila ditahan)."""
    return bool(jari_lurus(lm).all() and ibu_jari_terbuka(lm))


def posisi_kursor(lm) -> tuple[float, float]:
    """Titik 5 (pangkal telunjuk) dalam koordinat gambar ternormalisasi [0, 1]."""
    a = _cek(lm)
    return float(a[PANGKAL_TELUNJUK, 0]), float(a[PANGKAL_TELUNJUK, 1])
