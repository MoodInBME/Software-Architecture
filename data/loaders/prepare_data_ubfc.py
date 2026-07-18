"""Prapemrosesan UBFC-rPPG -> satu file .pt per subjek.

Menyimpan dict:
    {"video": Tensor(C,T,H,W) float32 [0,1],
     "ppg":   Tensor(T,)  (gelombang PPG diresample ke jumlah frame wajah),
     "hr_mean": float, "fps": float, "subject": str}

Perbaikan dari versi awal (lihat docs/CODE_REVIEW_data_layer.md):
- MENYIMPAN ground-truth PPG (versi lama hanya menyimpan tensor video dummy).
- Deteksi + crop wajah nyata via MediaPipe (bukan torch.randn).
- Path via argumen CLI (bukan hardcode /content/...). Tanpa token.
- Nama file aman (tidak memakai split('.')[0]).
"""
import os
import glob
import argparse

import numpy as np
import cv2
import torch

from dataset_ubfc_loader import load_ubfc_ground_truth

try:
    import mediapipe as mp
except ImportError:
    mp = None


def _facemesh():
    if mp is None:
        raise ImportError("mediapipe belum terpasang: pip install mediapipe==0.10.14")
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )


def _bbox(lms, w, h, margin=0.15):
    xs = np.array([p.x for p in lms]) * w
    ys = np.array([p.y for p in lms]) * h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    return (int(max(0, x0 - mw)), int(max(0, y0 - mh)),
            int(min(w, x1 + mw)), int(min(h, y1 + mh)))


def process_subject(vid_path, gt_path, size=(128, 128)):
    mesh = _facemesh()
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue
            h, w = rgb.shape[:2]
            x0, y0, x1, y1 = _bbox(res.multi_face_landmarks[0].landmark, w, h)
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, size).astype(np.float32) / 255.0
            frames.append(crop)
    finally:
        cap.release()
        mesh.close()

    if len(frames) == 0:
        return None

    video = np.transpose(np.asarray(frames), (3, 0, 1, 2))  # (C,T,H,W)
    n_frames = video.shape[1]
    ppg, hr, _ = load_ubfc_ground_truth(gt_path)
    # resample PPG agar panjangnya sama dengan jumlah frame wajah tersimpan
    ppg_rs = np.interp(
        np.linspace(0, len(ppg) - 1, n_frames), np.arange(len(ppg)), ppg
    ).astype(np.float32)
    return {
        "video": torch.from_numpy(np.ascontiguousarray(video)),
        "ppg": torch.from_numpy(ppg_rs),
        "hr_mean": float(np.mean(hr)) if hr.size else float("nan"),
        "fps": float(fps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True,
                    help="folder berisi subjectN/{vid.avi,ground_truth.txt}")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--video_name", default="vid.avi")
    ap.add_argument("--gt_name", default="ground_truth.txt")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    subjects = sorted(glob.glob(os.path.join(args.dataset_dir, "*")))
    saved = 0
    for sub in subjects:
        vid = os.path.join(sub, args.video_name)
        gt = os.path.join(sub, args.gt_name)
        if not (os.path.isfile(vid) and os.path.isfile(gt)):
            continue
        name = os.path.basename(os.path.normpath(sub))
        print(f"Memproses {name} ...")
        sample = process_subject(vid, gt)
        if sample is None:
            print(f"  dilewati (wajah tak terdeteksi): {name}")
            continue
        sample["subject"] = name
        torch.save(sample, os.path.join(args.output_dir, f"{name}.pt"))
        saved += 1
    print(f"Selesai. {saved} subjek tersimpan di {args.output_dir}")


if __name__ == "__main__":
    main()
