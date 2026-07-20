"""Ekstraksi & cache landmark wajah dari video/urutan gambar -> tensor (C, T, V).

Alur per klip:
    frame -> MediaPipe FaceMesh (468) -> subset_to_ibug68 (68) ->
    normalisasi spasial (pusatkan di hidung, skala jarak antar-mata) ->
    tumpuk (T, 68, 2) -> transpose (2, T, 68) -> simpan .npy + tulis manifest.

Hemat GPU: jalankan SEKALI, hasil di-cache; training membaca .npy (bukan video).

Contoh:
    python data/preprocessing/extract_landmarks_video.py \
        --input_root dataset/ckplus --layout ckplus \
        --output_dir data/prepared/ckplus --clip_len 150
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
    import mediapipe as mp
except ImportError:  # lingkungan build tanpa cv2/mediapipe
    cv2 = None
    mp = None

from data.preprocessing.landmark_map import subset_to_ibug68

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
NOSE_TIP_IBUG = 30      # ujung hidung pada indeks iBUG-68
LEYE_IBUG = 36          # sudut luar mata kanan
REYE_IBUG = 45          # sudut luar mata kiri


def _facemesh():
    if mp is None:
        raise RuntimeError("mediapipe/cv2 tidak terpasang di lingkungan ini.")
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )


def _frame_to_468(face_mesh, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x, p.y] for p in lm], dtype=np.float32)  # (468, 2)


def _normalize(pts68: np.ndarray) -> np.ndarray:
    center = pts68[NOSE_TIP_IBUG]
    scale = np.linalg.norm(pts68[REYE_IBUG] - pts68[LEYE_IBUG]) + 1e-6
    return (pts68 - center) / scale


def _iter_frames(clip_path: Path):
    if clip_path.is_dir():
        frames = sorted(p for p in clip_path.iterdir() if p.suffix.lower() in IMAGE_EXT)
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is not None:
                yield img
    else:
        cap = cv2.VideoCapture(str(clip_path))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()


def extract_clip(face_mesh, clip_path: Path, clip_len: int):
    seq = []
    last = None
    for frame in _iter_frames(clip_path):
        pts468 = _frame_to_468(face_mesh, frame)
        if pts468 is None:
            if last is None:
                continue
            pts68 = last
        else:
            pts68 = _normalize(subset_to_ibug68(pts468))
            last = pts68
        seq.append(pts68)
    if not seq:
        return None
    arr = np.stack(seq, axis=0)              # (T, 68, 2)
    t = arr.shape[0]
    if t >= clip_len:
        idx = np.linspace(0, t - 1, clip_len).round().astype(np.int64)
        arr = arr[idx]
    else:
        arr = np.concatenate([arr, np.repeat(arr[-1:], clip_len - t, axis=0)], axis=0)
    return arr.transpose(2, 0, 1).astype(np.float32)   # (C=2, T, V=68)


def discover_clips(input_root: Path, layout: str):
    clips = []
    if layout in ("ckplus", "emotion"):
        for emo_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
            emotion = emo_dir.name.lower()
            for clip in sorted(emo_dir.iterdir()):
                if clip.is_dir() or clip.suffix.lower() in VIDEO_EXT:
                    clips.append((clip, "emotion", emotion))
    elif layout == "affwild2":
        for clip in sorted(input_root.iterdir()):
            if clip.suffix.lower() in VIDEO_EXT:
                ann = clip.with_suffix(".txt")
                if not ann.exists():
                    continue
                vals = np.loadtxt(ann, delimiter=",", skiprows=1)
                v, a = float(np.mean(vals[:, 0])), float(np.mean(vals[:, 1]))
                clips.append((clip, "va", [v, a]))
    else:
        raise ValueError(f"layout tak dikenal: {layout}")
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--layout", default="ckplus", choices=["ckplus", "emotion", "affwild2"])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--clip_len", type=int, default=150)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "landmarks").mkdir(parents=True, exist_ok=True)
    clips = discover_clips(Path(args.input_root), args.layout)
    print(f"Ditemukan {len(clips)} klip.")

    face_mesh = _facemesh()
    manifest = []
    for i, (clip_path, label_type, value) in enumerate(clips):
        arr = extract_clip(face_mesh, clip_path, args.clip_len)
        if arr is None:
            print(f"[skip] tak ada wajah terdeteksi: {clip_path}")
            continue
        rel = f"landmarks/{i:05d}.npy"
        np.save(out_dir / rel, arr)
        manifest.append({"path": rel, "label_type": label_type, "value": value})
    face_mesh.close()

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Selesai. {len(manifest)} klip -> {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
