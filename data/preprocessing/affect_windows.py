"""Utilitas jendela landmark untuk dataset afek (AFEW-VA, DISFA) -> klip (C, T, V).

Dipakai bersama `data/loaders/afew_va.py` dan `data/loaders/disfa.py`. Semua fungsi di
sini murni NumPy (tanpa mediapipe, cv2, atau torch) agar bisa diuji di mana saja.

Keputusan rancangan:
  * Laju frame disamakan ke laju frame cermin (`target_fps`, bawaan 30) dengan
    interpolasi linear terhadap waktu, karena DISFA direkam 20 fps sedangkan jendela
    proposal adalah T sekitar 90 frame (3 detik) pada 30 fps.
  * Frame tanpa landmark (deteksi gagal) diisi dari frame valid terdekat (hold),
    sama dengan `extract_landmarks_video._landmarks_from_frames`, dan jumlah frame
    isian dicatat per klip (`frac_filled`) agar klip yang terlalu banyak isian bisa
    dibuang saat pelatihan.
  * Label kontinu dirata-rata per jendela. Nilai label tidak valid diberi NaN dan
    diabaikan dalam rerata; bila satu jendela tidak punya label valid, label = NaN
    dan mask = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Indeks sudut mata luar pada skema iBUG-68 (0-based): 36 (kanan subjek) dan 45 (kiri subjek).
OUTER_EYE = (36, 45)


def normalize_landmarks_seq(seq: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalisasi per frame: pusat ke centroid, skala dibagi jarak sudut mata luar.

    Sama dengan `landmark_extraction_mediapipe.normalize_landmarks`, diterapkan pada
    deret (T, 68, 2). Frame NaN dibiarkan NaN.
    """
    x = np.asarray(seq, dtype=np.float32)[..., :2].copy()
    if x.ndim != 3 or x.shape[1] != 68:
        raise ValueError(f"Diharapkan (T, 68, 2), dapat {x.shape}")
    x -= np.nanmean(x, axis=1, keepdims=True)
    io = np.linalg.norm(x[:, OUTER_EYE[0]] - x[:, OUTER_EYE[1]], axis=-1)
    x /= (io + eps)[:, None, None]
    return x


def fill_missing(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isi frame NaN dengan frame valid terdekat (maju, lalu awal dari valid pertama).

    Return (deret terisi, mask_valid (T,) bool). Galat bila tidak ada frame valid.
    """
    x = np.asarray(seq, dtype=np.float32).copy()
    valid = ~np.isnan(x).any(axis=(1, 2))
    if not valid.any():
        raise ValueError("Tidak ada frame dengan landmark valid")
    last = None
    for i in range(len(x)):
        if valid[i]:
            last = x[i]
        elif last is not None:
            x[i] = last
    first = int(np.argmax(valid))
    x[:first] = x[first]
    return x, valid


def resample_time(values: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Interpolasi linear sepanjang sumbu 0 dari src_fps ke dst_fps.

    Berlaku untuk deret landmark (T, V, C) maupun label (T,) atau (T, K). NaN pada
    label dipertahankan: titik hasil yang tetangga kiri atau kanannya NaN menjadi NaN.
    """
    v = np.asarray(values, dtype=np.float64)
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError("fps harus positif")
    n = v.shape[0]
    if n == 0 or abs(src_fps - dst_fps) < 1e-9:
        return v.astype(np.float32)
    t_src = np.arange(n) / src_fps
    n_dst = int(np.floor(t_src[-1] * dst_fps)) + 1
    t_dst = np.arange(n_dst) / dst_fps
    pos = t_dst * src_fps
    i0 = np.clip(np.floor(pos).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    w = (pos - i0).reshape((-1,) + (1,) * (v.ndim - 1))
    with np.errstate(invalid="ignore"):
        out = np.where(w == 0, v[i0], (1 - w) * v[i0] + w * v[i1])
    return out.astype(np.float32)


def chunk_indices(n_frames: int, clip_len: int, stride: int) -> list[tuple[int, int]]:
    """Sama dengan `extract_landmarks_video._chunk_indices` (jendela terakhir dirapatkan ke ujung)."""
    if clip_len <= 0 or stride <= 0:
        raise ValueError("clip_len dan stride harus positif")
    if n_frames <= 0:
        return []
    if n_frames <= clip_len:
        return [(0, n_frames)]
    idx = []
    start = 0
    while start + clip_len <= n_frames:
        idx.append((start, start + clip_len))
        start += stride
    if idx[-1][1] < n_frames:
        idx.append((n_frames - clip_len, n_frames))
    return idx


def window_mean(labels: np.ndarray, s: int, e: int) -> tuple[np.ndarray, np.ndarray]:
    """Rerata label pada [s, e) mengabaikan NaN. Return (nilai, mask) per kolom."""
    seg = np.asarray(labels, dtype=np.float32)[s:e]
    if seg.ndim == 1:
        seg = seg[:, None]
    ok = ~np.isnan(seg)
    cnt = ok.sum(axis=0)
    total = np.where(ok, seg, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
    return mean.astype(np.float32), (cnt > 0).astype(np.float32)


@dataclass
class ClipRecord:
    """Satu klip siap simpan. `landmarks` berbentuk (C=2, T, V=68)."""

    name: str
    landmarks: np.ndarray
    subject: str
    source: str
    valence: float = float("nan")
    arousal: float = float("nan")
    va_mask: float = 0.0
    au: list[float] = field(default_factory=list)
    au_mask: list[float] = field(default_factory=list)
    frac_filled: float = 0.0

    def manifest_entry(self, file_rel: str) -> dict:
        def f(x):
            return (
                None
                if x is None or (isinstance(x, float) and np.isnan(x))
                else float(x)
            )

        return {
            "file": file_rel,
            "valence": f(self.valence),
            "arousal": f(self.arousal),
            "va_mask": float(self.va_mask),
            "au": [f(v) for v in self.au],
            "au_mask": [float(m) for m in self.au_mask],
            "subject": self.subject,
            "source": self.source,
            "frac_filled": round(float(self.frac_filled), 4),
        }


def make_clips(
    landmarks: np.ndarray,
    valid: np.ndarray,
    name: str,
    subject: str,
    source: str,
    clip_len: int,
    stride: int,
    va: np.ndarray | None = None,
    au: np.ndarray | None = None,
    n_au: int = 0,
) -> list[ClipRecord]:
    """Potong deret (T, 68, 2) beserta label per frame menjadi ClipRecord.

    va: (T, 2) dalam [-1, 1] atau None. au: (T, K) dalam [0, 1] atau None (K = n_au).
    """
    T = landmarks.shape[0]
    for arr, nm in ((va, "va"), (au, "au")):
        if arr is not None and arr.shape[0] != T:
            raise ValueError(f"Panjang {nm} {arr.shape[0]} berbeda dengan landmark {T}")
    out = []
    for k, (s, e) in enumerate(chunk_indices(T, clip_len, stride)):
        rec = ClipRecord(
            name=f"{name}_{k:03d}",
            landmarks=np.ascontiguousarray(landmarks[s:e].transpose(2, 0, 1)).astype(
                np.float32
            ),
            subject=subject,
            source=source,
            frac_filled=float(1.0 - valid[s:e].mean()),
        )
        if va is not None:
            m, mk = window_mean(va, s, e)
            if mk.all():
                rec.valence, rec.arousal, rec.va_mask = float(m[0]), float(m[1]), 1.0
        if au is not None:
            m, mk = window_mean(au, s, e)
            rec.au = [float(x) for x in m]
            rec.au_mask = [float(x) for x in mk]
        elif n_au:
            rec.au = [float("nan")] * n_au
            rec.au_mask = [0.0] * n_au
        if rec.va_mask == 0.0 and not any(rec.au_mask):
            continue  # klip tanpa label apa pun tidak berguna untuk pelatihan
        out.append(rec)
    return out


def save_clips(records: list[ClipRecord], clips_dir: str, manifest: list) -> None:
    """Simpan klip sebagai .pt (format sama dengan ExpressionLandmarkDataset) dan isi manifest."""
    import os

    import torch

    os.makedirs(clips_dir, exist_ok=True)
    for rec in records:
        fname = f"{rec.name}.pt"
        entry = rec.manifest_entry(os.path.join("clips", fname))
        payload = dict(entry)
        payload.pop("file")
        payload["landmarks"] = torch.from_numpy(rec.landmarks)
        torch.save(payload, os.path.join(clips_dir, fname))
        manifest.append(entry)
