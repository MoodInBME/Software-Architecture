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
                f"{path}: kolom {need!r} tidak ada (jalankan FeatureExtraction dengan -2Dfp "
                "dan -aus; CSV FaceLandmarkImg dibaca dengan read_openface_image_csv)"
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


def read_openface_image_csv(path: str) -> dict:
    """Pembaca CSV OpenFace ``FaceLandmarkImg`` (satu gambar, satu baris per wajah).

    Berbeda dari FeatureExtraction, CSV gambar tidak punya kolom ``frame`` dan
    ``success``; kolom ``face`` dan ``confidence`` bisa ada. Return dict:
    ``landmarks`` (W, 68, 2) piksel, ``confidence`` (W,) (NaN bila kolom tidak ada),
    ``au`` {nomor: (W,)} intensitas mentah 0..5 (ubah dengan
    ``data.preprocessing.au_scale.ke_satuan`` sebelum dibandingkan dengan kepala AU),
    dan ``n_wajah``. CSV tanpa baris (tidak ada wajah) memberi ``n_wajah`` = 0.

    Contoh perintah (di luar repo; keluaran ke D:\\moodin_runs\\):
        FaceLandmarkImg.exe -fdir <folder gambar> -2Dfp -aus -out_dir <folder keluaran>
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            raise ValueError(f"{path}: CSV kosong tanpa header") from None
        rows = [r for r in reader if r]
    col = {h: i for i, h in enumerate(header)}
    for need in ("x_0", "y_67"):
        if need not in col:
            raise KeyError(f"{path}: kolom {need!r} tidak ada (jalankan dengan -2Dfp)")
    if not rows:
        return {
            "landmarks": np.zeros((0, 68, 2), np.float32),
            "confidence": np.zeros(0, np.float32),
            "au": {},
            "n_wajah": 0,
        }
    arr = np.asarray([[float(v) for v in r] for r in rows], dtype=np.float32)
    xs = arr[:, [col[f"x_{i}"] for i in range(68)]]
    ys = arr[:, [col[f"y_{i}"] for i in range(68)]]
    conf = arr[:, col["confidence"]] if "confidence" in col else np.full(len(arr), np.nan)
    aus = {
        int(h[2:4]): arr[:, col[h]].copy()
        for h in header
        if h.startswith("AU") and h.endswith("_r")
    }
    return {
        "landmarks": np.stack([xs, ys], axis=-1),
        "confidence": conf,
        "au": aus,
        "n_wajah": int(len(arr)),
    }


def best_face(record: dict, min_confidence: float = 0.0):
    """Landmark (68, 2) wajah dengan confidence tertinggi, atau None bila tidak ada/terlalu rendah.

    Bila kolom confidence tidak ada, wajah pertama dipakai.
    """
    if record["n_wajah"] == 0:
        return None
    conf = record["confidence"]
    if np.all(np.isnan(conf)):
        return record["landmarks"][0]
    k = int(np.nanargmax(conf))
    return record["landmarks"][k] if conf[k] >= min_confidence else None
