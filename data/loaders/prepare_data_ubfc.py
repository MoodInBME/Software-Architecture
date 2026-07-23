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

# Batasi thread OpenCV: mencegah OpenCV dan thread pool internal
# TFLite/XNNPACK (dipakai MediaPipe) berebut resource, yang bisa
# memicu crash heap native ("malloc_consolidate: invalid chunk size").
cv2.setNumThreads(1)

from dataset_ubfc_loader import load_ubfc_ground_truth

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp = None

FACE_LANDMARKER_MODEL_PATH = os.environ.get(
    "FACE_LANDMARKER_MODEL_PATH", "/content/face_landmarker.task"
)


def _facemesh():
    if mp is None:
        raise ImportError("mediapipe belum terpasang: pip install mediapipe")
    if not os.path.isfile(FACE_LANDMARKER_MODEL_PATH):
        raise FileNotFoundError(
            f"Model {FACE_LANDMARKER_MODEL_PATH} tidak ditemukan. Unduh dulu dengan:\n"
            "wget -O /content/face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        )
    base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def _bbox(lms, w, h, margin=0.15):
    xs = np.array([p.x for p in lms]) * w
    ys = np.array([p.y for p in lms]) * h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    return (int(max(0, x0 - mw)), int(max(0, y0 - mh)),
            int(min(w, x1 + mw)), int(min(h, y1 + mh)))


def process_subject(vid_path, gt_path, size=(128, 128)):
    detector = _facemesh()
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_ms = 1000.0 / fps
    frames = []
    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * frame_interval_ms)
            res = detector.detect_for_video(mp_image, timestamp_ms)
            frame_idx += 1
            if not res.face_landmarks:
                continue
            h, w = rgb.shape[:2]
            x0, y0, x1, y1 = _bbox(res.face_landmarks[0], w, h)
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, size).astype(np.float32) / 255.0
            frames.append(crop)
    finally:
        cap.release()
        detector.close()

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
