"""Pembaca berkas UBFC-Phys dan pemetaan tugas ke tingkat stres.

Isi per subjek sN (diperiksa pada s1, 18 November 2026): ``vid_sN_T1..T3.avi``,
``bvp_sN_T1..T3.csv`` (64 Hz, satu nilai per baris), ``eda_sN_T1..T3.csv``, ``info_sN.txt``
(jenis kelamin, kelompok ``test`` atau ``ctrl``, tanggal, jam), dan
``selfReportedAnx_sN.csv``.

Pemetaan ke tiga kelas ordinal ADALAH ASUMSI penelitian ini, bukan label dataset:
T1 (istirahat) Normal; T2 dan T3 kelompok ctrl Stres Ringan; T2 dan T3 kelompok test Stres
Akut. Kelompok ctrl mengerjakan versi tugas yang lebih ringan (Meziati Sabour dkk., 2023),
tetapi tingkat stres yang benar-benar dialami tidak diukur langsung; ``selfReportedAnx``
bisa dipakai untuk memeriksa asumsi ini.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

FS_BVP = 64.0
TUGAS = ("T1", "T2", "T3")
KELAS_STRES = ("Normal", "Stres Ringan", "Stres Akut")
PETA_ASUMSI = {
    ("T1", "ctrl"): 0,
    ("T1", "test"): 0,
    ("T2", "ctrl"): 1,
    ("T3", "ctrl"): 1,
    ("T2", "test"): 2,
    ("T3", "test"): 2,
}


def berkas_sinyal(sid: str) -> list[str]:
    """Nama berkas kecil (tanpa video) yang dibutuhkan untuk satu subjek."""
    out = [f"info_{sid}.txt", f"selfReportedAnx_{sid}.csv"]
    for t in TUGAS:
        out += [f"bvp_{sid}_{t}.csv", f"eda_{sid}_{t}.csv"]
    return out


def baca_kelompok(path) -> str:
    """'test' atau 'ctrl' dari info_sN.txt. Galat bila tidak ada atau keduanya muncul."""
    teks = Path(path).read_text(encoding="utf-8", errors="replace").lower()
    ada = {k for k in ("test", "ctrl") if re.search(rf"\b{k}\b", teks)}
    if len(ada) != 1:
        raise ValueError(
            f"{path}: kelompok tidak bisa ditentukan (ditemukan {sorted(ada) or 'tidak ada'}). "
            "Buka berkas dan periksa baris kelompok."
        )
    return ada.pop()


def baca_bvp(path) -> np.ndarray:
    v = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=1).ravel()
    if v.size == 0 or not np.isfinite(v).all():
        raise ValueError(f"{path}: BVP kosong atau memuat nilai tidak hingga")
    return v


def label_tugas(tugas: str, kelompok: str, peta: dict | None = None) -> int:
    peta = PETA_ASUMSI if peta is None else peta
    try:
        return peta[(tugas, kelompok)]
    except KeyError:
        raise ValueError(
            f"Tidak ada label untuk tugas {tugas!r} kelompok {kelompok!r}"
        ) from None


def subjek_lengkap(folder, sid: str, min_detik: float = 170.0) -> tuple[bool, str]:
    """(lengkap, alasan). Lengkap bila info terbaca dan BVP ketiga tugas cukup panjang."""
    d = Path(folder)
    try:
        baca_kelompok(d / f"info_{sid}.txt")
    except (OSError, ValueError) as e:
        return False, f"info: {e}"
    for t in TUGAS:
        p = d / f"bvp_{sid}_{t}.csv"
        try:
            n = baca_bvp(p).size
        except (OSError, ValueError) as e:
            return False, f"bvp {t}: {e}"
        if n / FS_BVP < min_detik:
            return False, f"bvp {t}: {n / FS_BVP:.0f} s < {min_detik:.0f} s"
    return True, "lengkap"


def pilih_seimbang(
    kelompok: dict[str, str], n_per_kelompok: int, seed: int
) -> list[str]:
    """Pilih n subjek dari tiap kelompok secara acak dengan seed tetap, urut nomor subjek."""
    rng = np.random.default_rng(seed)
    pilihan = []
    for k in ("test", "ctrl"):
        calon = sorted((s for s, g in kelompok.items() if g == k), key=_nomor)
        if len(calon) < n_per_kelompok:
            raise ValueError(
                f"Kelompok {k} hanya punya {len(calon)} subjek lengkap; butuh {n_per_kelompok}."
            )
        pilihan += list(rng.choice(calon, size=n_per_kelompok, replace=False))
    return sorted(pilihan, key=_nomor)


def _nomor(sid: str) -> int:
    m = re.search(r"\d+", sid)
    return int(m.group()) if m else 0


def jendela(
    n_sampel: int, fs: float, panjang_s: float, geser_s: float, mulai_s: float = 0.0
):
    """Indeks (awal, akhir) jendela penuh dalam sinyal sepanjang n_sampel."""
    p, g = int(round(panjang_s * fs)), int(round(geser_s * fs))
    a = int(round(mulai_s * fs))
    out = []
    while a + p <= n_sampel:
        out.append((a, a + p))
        a += g
    return out
