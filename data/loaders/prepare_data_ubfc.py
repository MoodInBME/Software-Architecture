"""
Prapemrosesan UBFC-rPPG langsung dari file .zip -> satu file .pt per subjek.

Berbeda dari versi folder biasa:
- Tidak perlu unzip manual. Skrip membaca isi zip, lalu untuk tiap subjek
  hanya mengekstrak vid.avi + ground_truth.txt ke folder sementara
  (tempfile), memprosesnya, lalu menghapus file sementara itu.
- Struktur di dalam zip diasumsikan: <root>/<subject>/vid.avi
  dan <root>/<subject>/ground_truth.txt (root boleh ada atau tidak,
  skrip mendeteksi otomatis berdasarkan lokasi vid.avi).

Menyimpan dict:
    {"video": Tensor(C,T,H,W) float32 [0,1],
     "ppg":   Tensor(T,),
     "hr_mean": float, "fps": float, "subject": str}
"""
import os
import argparse
import tempfile
import zipfile
from collections import defaultdict

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
    ppg_rs = np.interp(
        np.linspace(0, len(ppg) - 1, n_frames), np.arange(len(ppg)), ppg
    ).astype(np.float32)
    return {
        "video": torch.from_numpy(np.ascontiguousarray(video)),
        "ppg": torch.from_numpy(ppg_rs),
        "hr_mean": float(np.mean(hr)) if hr.size else float("nan"),
        "fps": float(fps),
    }


def _find_subjects_in_zip(zf, video_name, gt_name):
    """Kelompokkan entry zip berdasarkan folder subjek yang berisi
    video_name DAN gt_name sekaligus."""
    by_dir = defaultdict(set)
    for info in zf.infolist():
        if info.is_dir():
            continue
        dirname = os.path.dirname(info.filename)
        base = os.path.basename(info.filename)
        if base in (video_name, gt_name):
            by_dir[dirname].add(base)

    subjects = {}
    for dirname, files in by_dir.items():
        if video_name in files and gt_name in files:
            subject_name = os.path.basename(dirname.rstrip("/")) or dirname
            subjects[subject_name] = dirname
    return subjects  # {subject_name: dir_path_inside_zip}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_zip", required=True,
                    help="path ke file .zip dataset UBFC-rPPG")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--video_name", default="vid.avi")
    ap.add_argument("--gt_name", default="ground_truth.txt")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with zipfile.ZipFile(args.dataset_zip, "r") as zf:
        subjects = _find_subjects_in_zip(zf, args.video_name, args.gt_name)
        print(f"Ditemukan {len(subjects)} subjek di dalam zip.")
        if len(subjects) == 0:
            # bantu debug: tampilkan beberapa entry pertama di zip
            print("Contoh isi zip (20 entry pertama):")
            for n in zf.namelist()[:20]:
                print(" ", n)
            return

        saved = 0
        for name, dirname in sorted(subjects.items()):
            print(f"Memproses {name} ...")
            with tempfile.TemporaryDirectory() as tmp:
                vid_tmp = os.path.join(tmp, args.video_name)
                gt_tmp = os.path.join(tmp, args.gt_name)
                with zf.open(f"{dirname}/{args.video_name}") as src, open(vid_tmp, "wb") as dst:
                    dst.write(src.read())
                with zf.open(f"{dirname}/{args.gt_name}") as src, open(gt_tmp, "wb") as dst:
                    dst.write(src.read())

                sample = process_subject(vid_tmp, gt_tmp)
                # vid_tmp dan gt_tmp otomatis terhapus saat blok `with` selesai

            if sample is None:
                print(f"  dilewati (wajah tak terdeteksi): {name}")
                continue
            sample["subject"] = name
            torch.save(sample, os.path.join(args.output_dir, f"{name}.pt"))
            saved += 1

    print(f"Selesai. {saved} subjek tersimpan di {args.output_dir}")


if __name__ == "__main__":
    main()
