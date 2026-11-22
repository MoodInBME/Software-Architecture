"""DISFA (Mavadati dkk., IEEE Trans. Affective Computing 2013) -> klip landmark berlabel intensitas AU.

Akses: formulir perjanjian di laman pengelola (Mohammad H. Mahoor, University of Denver).
DISFA+ memakai perjanjian terpisah dan struktur folder berbeda, sehingga belum dibaca modul
ini; tambahkan parser setelah README DISFA+ diterima. Data TIDAK boleh masuk repo.

Isi yang dipakai [Medium, cocokkan dengan README setelah akses diberikan]:
    <root>/ActionUnit_Labels/SN001/SN001_au1.txt   # baris "frame,intensitas", frame mulai 1
    <root>/Videos_LeftCamera/LeftVideoSN001_comp.avi  # 20 fps
12 AU berlabel intensitas 0..5: 1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26. AU7 TIDAK ada.

Landmark: titik bawaan DISFA berjumlah 66, sedangkan graf memakai 68. Modul ini
mengekstrak ulang landmark dengan ekstraktor cermin (MediaPipe -> iBUG-68), sehingga
masalah 66 titik tidak muncul dan domain masukan sama dengan saat inferensi.

Contoh:
    python data/loaders/disfa.py --root ../datasets/DISFA --output_dir ../prepared/disfa
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.preprocessing.affect_windows import (
    fill_missing,
    make_clips,
    resample_time,
    save_clips,
)  # noqa: E402

DISFA_AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
from data.preprocessing.au_scale import AU_MAX as MAX_INTENSITY  # noqa: E402
SRC_FPS = 20.0


def read_au_file(path: str, n_frames: int | None = None) -> np.ndarray:
    """Baca satu berkas AU -> (F,) intensitas 0..5. Frame tanpa baris diberi NaN."""
    rows = {}
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 2:
                raise ValueError(f"{path}:{ln} format tidak dikenal: {line!r}")
            fr, val = int(float(parts[0])), float(parts[1])
            if not 0 <= val <= MAX_INTENSITY:
                raise ValueError(f"{path}:{ln} intensitas {val} di luar 0..5")
            rows[fr] = val
    if not rows:
        raise ValueError(f"{path} kosong")
    n = n_frames or max(rows)
    out = np.full(n, np.nan, np.float32)
    for fr, val in rows.items():
        if 1 <= fr <= n:
            out[fr - 1] = val
    return out


def read_subject_labels(label_dir: str, aus=DISFA_AUS) -> np.ndarray:
    """Return (F, K) intensitas ternormalisasi 0..1 untuk satu subjek (urutan = aus)."""
    sid = Path(label_dir).name
    cols = []
    for au in aus:
        cand = glob.glob(os.path.join(label_dir, f"{sid}_au{au}.txt"))
        if len(cand) != 1:
            raise FileNotFoundError(
                f"Label AU{au} untuk {sid} tidak ditemukan tepat satu di {label_dir}"
            )
        cols.append(read_au_file(cand[0]))
    n = min(len(c) for c in cols)
    return np.stack([c[:n] for c in cols], axis=1) / MAX_INTENSITY


def find_video(root: str, sid: str) -> str:
    hits = glob.glob(os.path.join(root, "**", f"*{sid}*.avi"), recursive=True)
    left = [h for h in hits if "left" in os.path.basename(h).lower()]
    hits = left or hits
    if len(hits) != 1:
        raise FileNotFoundError(f"Video {sid}: ditemukan {len(hits)} kandidat")
    return hits[0]


def video_landmarks(path: str, extractor, max_frames: int | None = None) -> np.ndarray:
    """Ekstrak landmark frame demi frame tanpa menyimpan seluruh video di memori."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"Video tidak dapat dibuka: {path}")
    out = []
    try:
        while max_frames is None or len(out) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            pts = extractor.extract(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            out.append(np.full((68, 2), np.nan, np.float32) if pts is None else pts)
    finally:
        cap.release()
    return np.stack(out, axis=0)


def prepare(root, output_dir, clip_len, stride, dst_fps, max_filled, extractor):
    manifest = []
    label_dirs = sorted(glob.glob(os.path.join(root, "ActionUnit_Labels", "SN*")))
    if not label_dirs:
        raise FileNotFoundError(f"Folder ActionUnit_Labels/SN* tidak ada di {root}")
    for ld in label_dirs:
        sid = Path(ld).name
        au = read_subject_labels(ld)
        lm = video_landmarks(find_video(root, sid), extractor, max_frames=len(au))
        n = min(len(au), len(lm))
        if n < len(au):
            print(
                f"Peringatan {sid}: video {len(lm)} frame, label {len(au)} frame; dipotong ke {n}"
            )
        seq, valid = fill_missing(lm[:n])
        au = au[:n]
        seq = resample_time(seq, SRC_FPS, dst_fps)
        valid = resample_time(valid.astype(np.float32), SRC_FPS, dst_fps) > 0.999
        au = resample_time(au, SRC_FPS, dst_fps)
        recs = make_clips(
            seq, valid, f"disfa_{sid}", sid, "disfa", clip_len, stride, au=au
        )
        recs = [r for r in recs if r.frac_filled <= max_filled]
        save_clips(recs, os.path.join(output_dir, "clips"), manifest)
        print(f"{sid}: {len(recs)} klip")
    meta = {"au_list": DISFA_AUS, "intensity_scale": MAX_INTENSITY, "fps": dst_fps}
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return manifest


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--clip_len", type=int, default=90)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--dst_fps", type=float, default=30.0)
    ap.add_argument("--max_filled", type=float, default=0.2)
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    if out.is_relative_to(_ROOT.resolve()):
        raise SystemExit("output_dir di dalam repo. Simpan klip turunan di luar repo.")
    os.makedirs(out, exist_ok=True)
    from data.preprocessing.landmark_extraction_mediapipe import (
        MediaPipeLandmarkExtractor,
    )

    extractor = MediaPipeLandmarkExtractor(static_image_mode=False)
    try:
        prepare(
            args.root,
            str(out),
            args.clip_len,
            args.stride,
            args.dst_fps,
            args.max_filled,
            extractor,
        )
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
