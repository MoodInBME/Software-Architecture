"""Pembaca keluaran OpenFace 2.0 FeatureExtraction (Baltrušaitis dkk., FG 2018).

Peran di MoodIn: pembanding luring saja, BUKAN bagian runtime cermin.
  * Intensitas AU OpenFace pada DISFA sebagai pembanding kepala AU (OpenFace dilatih
    antara lain dengan DISFA, jadi angka ini tidak boleh dibaca sebagai uji lintas data).
  * Landmark 68 titik OpenFace sebagai pembanding ekstraktor MediaPipe -> iBUG-68.
OpenFace 2.0 berlisensi akademik nonkomersial. Jangan tertukar dengan cmusatyalab/openface
(FaceNet, pengenalan identitas) yang justru memperkenalkan kembali biometrik.

Contoh perintah OpenFace (dijalankan terpisah, di luar repo):
    FeatureExtraction.exe -f LeftVideoSN001_comp.avi -2Dfp -aus -pose -out_dir out
"""

from __future__ import annotations

import csv

import numpy as np


# Pemetaan nomor AU ke kolom intensitas OpenFace ("AU01_r" dst.).
def au_col(au: int) -> str:
    return f"AU{au:02d}_r"


def read_openface_csv(path: str, min_confidence: float = 0.8) -> dict:
    """Return dict: frame (F,), success (F,) bool, landmarks (F, 68, 2) NaN bila gagal, au {nomor: (F,)}.

    Nama kolom OpenFace versi lama diawali spasi; kolom di-strip sebelum dipakai.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [r for r in reader if r]
    col = {h: i for i, h in enumerate(header)}
    for need in ("frame", "confidence", "success", "x_0", "y_67"):
        if need not in col:
            raise KeyError(
                f"{path}: kolom {need!r} tidak ada (jalankan dengan -2Dfp dan -aus)"
            )
    arr = np.asarray([[float(v) for v in r] for r in rows], dtype=np.float32)
    ok = (arr[:, col["success"]] > 0.5) & (arr[:, col["confidence"]] >= min_confidence)
    xs = arr[:, [col[f"x_{i}"] for i in range(68)]]
    ys = arr[:, [col[f"y_{i}"] for i in range(68)]]
    lm = np.stack([xs, ys], axis=-1)
    lm[~ok] = np.nan
    aus = {}
    for h in header:
        if h.startswith("AU") and h.endswith("_r"):
            v = arr[:, col[h]].copy()
            v[~ok] = np.nan
            aus[int(h[2:4])] = v
    return {
        "frame": arr[:, col["frame"]].astype(int),
        "success": ok,
        "landmarks": lm,
        "au": aus,
    }
