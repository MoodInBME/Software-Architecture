"""Pembagian data berbasis subjek (tanpa kebocoran identitas wajah antarsplit).

AFEW-VA: subjek = aktor (satu aktor dapat muncul di banyak klip). DISFA: subjek = SNxxx.
Pembagian deterministik dengan seed agar dapat diulang di laptop, Kaggle, dan Jetson.
"""

from __future__ import annotations

import random


def split_subjects(
    subjects, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> dict:
    subs = sorted(set(subjects))
    if len(subs) < 3:
        raise ValueError("Minimal 3 subjek untuk pembagian latih, validasi, dan uji")
    rng = random.Random(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    if n_test + n_val >= n:
        raise ValueError("Fraksi validasi dan uji terlalu besar untuk jumlah subjek")
    return {
        "test": sorted(subs[:n_test]),
        "val": sorted(subs[n_test : n_test + n_val]),
        "train": sorted(subs[n_test + n_val :]),
    }


def k_fold_subjects(subjects, k: int = 3, seed: int = 42) -> list[dict]:
    """Lipatan subjek-independen (misalnya 3 lipatan, protokol umum DISFA)."""
    subs = sorted(set(subjects))
    if k < 2 or k > len(subs):
        raise ValueError("k harus antara 2 dan jumlah subjek")
    rng = random.Random(seed)
    rng.shuffle(subs)
    folds = [sorted(subs[i::k]) for i in range(k)]
    return [
        {
            "test": folds[i],
            "train": sorted(s for j, f in enumerate(folds) if j != i for s in f),
        }
        for i in range(k)
    ]
