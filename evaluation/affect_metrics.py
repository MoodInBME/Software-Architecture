"""Metrik jalur biomekanis: NME landmark, ICC(3,1) dan MAE intensitas AU, CCC valensi-arousal.

NME: galat titik ke titik dirata-rata lalu dibagi jarak
sudut mata luar acuan (titik 36 dan 45 skema iBUG-68). ICC(3,1) adalah metrik baku untuk
intensitas AU pada DISFA (Shrout dan Fleiss 1979, model dua arah campuran, konsistensi).
"""

from __future__ import annotations

import numpy as np


def nme(pred: np.ndarray, gt: np.ndarray, outer_eye=(36, 45)) -> np.ndarray:
    """pred, gt: (N, 68, 2) dalam satuan sama. Return (N,) NME per frame; frame NaN diabaikan."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.shape[-2:] != (68, 2):
        raise ValueError(f"Bentuk {pred.shape} dan {gt.shape} harus (N, 68, 2)")
    d = np.linalg.norm(gt[:, outer_eye[0]] - gt[:, outer_eye[1]], axis=-1)
    err = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, err / d, np.nan)


def icc31(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ICC(3,1) dua penilai (acuan dan prediksi) atas N sasaran. NaN dibuang berpasangan."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    x = np.stack([a[ok], b[ok]], axis=1)
    n, k = x.shape
    if n < 2:
        return float("nan")
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((x - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err
    return float((ms_rows - ms_err) / denom) if denom > 0 else float("nan")


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance correlation coefficient (Lin 1989), varians populasi."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if a.size < 2:
        return float("nan")
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    denom = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")


def au_report(
    y_true: np.ndarray, y_pred: np.ndarray, au_list, scale: float = 5.0
) -> dict:
    """y_true, y_pred: (N, K) dalam 0..1. Laporan per AU pada skala asli 0..5."""
    out = {}
    for j, au in enumerate(au_list):
        t, p = y_true[:, j] * scale, y_pred[:, j] * scale
        ok = ~(np.isnan(t) | np.isnan(p))
        out[f"AU{au}"] = {
            "icc31": icc31(t, p),
            "mae": float(np.abs(t[ok] - p[ok]).mean()) if ok.any() else float("nan"),
            "n": int(ok.sum()),
        }
    vals = [v["icc31"] for v in out.values() if not np.isnan(v["icc31"])]
    out["rerata_icc31"] = float(np.mean(vals)) if vals else float("nan")
    return out
