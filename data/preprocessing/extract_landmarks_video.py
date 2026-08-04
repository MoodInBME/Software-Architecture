"""Ekstraksi + cache landmark wajah dari video/urutan gambar -> tensor (C,T,V).

Menghasilkan:
  <output_dir>/clips/<name>_<k>.pt  # dict {landmarks (2,T,68), valence, arousal, subject, source}
  <output_dir>/manifest.json        # daftar entri untuk ExpressionLandmarkDataset

Layout yang didukung:
  --layout ckplus   : CK+ (urutan gambar per-sekuens + label emosi kategorikal)
  --layout affwild2 : Aff-Wild2 (video + anotasi VA kontinu per-frame)

Contoh:
  python data/preprocessing/extract_landmarks_video.py \
      --input_root dataset/ckplus --layout ckplus \
      --output_dir data/prepared/ckplus --clip_len 150 --stride 15

Catatan: butuh mediapipe + opencv (cv2) + torch. Tidak dieksekusi di lingkungan
build; lulus py_compile untuk validasi sintaks.

DESAIN PENTING (belajar dari isu STC-rPPG): frame yang gagal deteksi wajah
TIDAK di-drop, melainkan diisi (hold) dari frame valid terdekat, supaya panjang
temporal & sinkronisasi label tetap terjaga.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.preprocessing.landmark_extraction_mediapipe import MediaPipeLandmarkExtractor

try:
    from models.st_gcn_graph_tran.labels import emotion_to_va as _emotion_to_va_fn
except Exception:
    _emotion_to_va_fn = None

# CK+ Emotion code (isi file .txt FACS) -> nama emosi
CKPLUS_CODE_TO_EMOTION = {
    0: "neutral", 1: "anger", 2: "contempt", 3: "disgust",
    4: "fear", 5: "happy", 6: "sad", 7: "surprise",
}

_FALLBACK_EMO_VA = {
    "neutral": (0.0, 0.0), "happy": (0.8, 0.5), "sad": (-0.7, -0.4),
    "anger": (-0.7, 0.7), "fear": (-0.6, 0.8), "disgust": (-0.6, 0.3),
    "surprise": (0.3, 0.8), "contempt": (-0.5, 0.2),
}


def _emotion_va(emo):
    if _emotion_to_va_fn is not None:
        try:
            return _emotion_to_va_fn(emo)
        except Exception:
            pass
    return _FALLBACK_EMO_VA.get(emo, (0.0, 0.0))


def _chunk_indices(n_frames, clip_len, stride):
    if n_frames <= 0:
        return []
    if n_frames <= clip_len:
        return [(0, n_frames)]
    idx = []
    start = 0
    while start + clip_len <= n_frames:
        idx.append((start, start + clip_len))
        start += stride
    if idx and idx[-1][1] < n_frames:
        idx.append((n_frames - clip_len, n_frames))
    return idx


def _landmarks_from_frames(frames_rgb, extractor):
    """frames_rgb: list (H,W,3) RGB. Return (T,68,2); frame gagal deteksi diisi
    dari tetangga valid (hold) agar temporal tetap sinkron. None bila semua gagal."""
    seq = []
    for fr in frames_rgb:
        seq.append(extractor.extract(fr))
    if not any(p is not None for p in seq):
        return None
    # forward-fill
    last = None
    for i in range(len(seq)):
        if seq[i] is not None:
            last = seq[i]
        elif last is not None:
            seq[i] = last.copy()
    # frame awal yang masih None -> pakai valid pertama
    first_valid = next(p for p in seq if p is not None)
    seq = [p if p is not None else first_valid.copy() for p in seq]
    return np.stack(seq, axis=0)  # (T,68,2)


def _save_clips(lands, v, a, subject, source, name, clip_len, stride, clips_dir, manifest):
    import torch
    total = lands.shape[0]
    for k, (s, e) in enumerate(_chunk_indices(total, clip_len, stride)):
        clip = lands[s:e]  # (t,68,2)
        x = torch.from_numpy(np.ascontiguousarray(clip.transpose(2, 0, 1))).float()  # (2,t,68)
        fname = f"{name}_{k:03d}.pt"
        torch.save(
            {"landmarks": x, "valence": float(v), "arousal": float(a),
             "subject": subject, "source": source},
            os.path.join(clips_dir, fname),
        )
        manifest.append({"file": os.path.join("clips", fname), "valence": float(v),
                         "arousal": float(a), "subject": subject, "source": source})


def process_ckplus(input_root, output_dir, clip_len, stride, extractor):
    import cv2
    import collections
    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    manifest = []
    skipped = 0

    # Cek format: apakah ada folder "anger", "happy", dll (Kaggle CK+48)
    subdirs = [d.lower() for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    is_kaggle = any(emo in subdirs for emo in ["anger", "happy", "sadness", "surprise"])

    if is_kaggle:
        print("[ckplus] Terdeteksi format Kaggle CK+48 (folder emosi).")
        seqs = collections.defaultdict(list)
        for emo_folder in os.listdir(input_root):
            emo_path = os.path.join(input_root, emo_folder)
            if not os.path.isdir(emo_path): continue
            emo = emo_folder.lower()
            if emo == "sadness": emo = "sad"
            if emo not in _FALLBACK_EMO_VA: continue
            
            for fname in os.listdir(emo_path):
                if not fname.endswith(('.png', '.jpg')): continue
                parts = fname.split('_')
                if len(parts) >= 2:
                    subj_seq = f"{parts[0]}_{parts[1]}"
                    seqs[(subj_seq, emo)].append(os.path.join(emo_path, fname))
                    
        for (subj_seq, emo), frame_paths in seqs.items():
            frame_paths = sorted(frame_paths)
            v, a = _emotion_va(emo)
            frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in frame_paths]
            lands = _landmarks_from_frames(frames, extractor)
            if lands is None:
                skipped += 1
                continue
            
            # CK+48 Kaggle hanya berisi beberapa frame puncak.
            # Pad (ulangi frame terakhir) hingga mencapai clip_len agar ST-GCN bisa jalan.
            if lands.shape[0] < clip_len:
                pad_len = clip_len - lands.shape[0]
                pad_lands = np.repeat(lands[-1:], pad_len, axis=0)
                lands = np.concatenate([lands, pad_lands], axis=0)
                
            _save_clips(lands, v, a, subj_seq, "ckplus_kaggle", subj_seq,
                        clip_len, stride, clips_dir, manifest)
        
        print(f"[ckplus] sekuens dilewati (tanpa wajah): {skipped}")
        return manifest

    # Format asli CK+ (images/ dan Emotion/)
    images_root = input_root
    for cand in ("cohn-kanade-images", "images"):
        p = os.path.join(input_root, cand)
        if os.path.isdir(p):
            images_root = p
            break
    emotion_root = None
    for cand in ("Emotion", "emotion", "Emotion_labels"):
        p = os.path.join(input_root, cand)
        if os.path.isdir(p):
            emotion_root = p
            break

    subjects = sorted(d for d in os.listdir(images_root)
                      if os.path.isdir(os.path.join(images_root, d)))
    for subj in subjects:
        subj_dir = os.path.join(images_root, subj)
        for seq in sorted(os.listdir(subj_dir)):
            seq_dir = os.path.join(subj_dir, seq)
            if not os.path.isdir(seq_dir):
                continue
            frame_paths = sorted(
                glob.glob(os.path.join(seq_dir, "*.png"))
                + glob.glob(os.path.join(seq_dir, "*.jpg"))
            )
            if len(frame_paths) < 2:
                continue
            emo = None
            if emotion_root:
                lbl = glob.glob(os.path.join(emotion_root, subj, seq, "*.txt"))
                if lbl:
                    try:
                        with open(lbl[0]) as f:
                            emo = CKPLUS_CODE_TO_EMOTION.get(int(float(f.read().strip())))
                    except Exception:
                        emo = None
            if emo is None:
                skipped += 1
                continue  # tanpa label VA tak bisa dilatih (CK+ neutral sering tak berlabel)
            v, a = _emotion_va(emo)
            frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in frame_paths]
            lands = _landmarks_from_frames(frames, extractor)
            if lands is None:
                skipped += 1
                continue
            
            if lands.shape[0] < clip_len:
                pad_len = clip_len - lands.shape[0]
                pad_lands = np.repeat(lands[-1:], pad_len, axis=0)
                lands = np.concatenate([lands, pad_lands], axis=0)

            _save_clips(lands, v, a, subj, "ckplus", f"{subj}_{seq}",
                        clip_len, stride, clips_dir, manifest)
    print(f"[ckplus] sekuens dilewati (tanpa label/wajah): {skipped}")
    return manifest


def _read_affwild_ann(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("valence"):
                continue
            parts = line.replace(";", ",").split(",")
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except Exception:
                continue
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 2), np.float32)


def process_affwild2(input_root, output_dir, clip_len, stride, extractor):
    import cv2
    import torch
    videos = (glob.glob(os.path.join(input_root, "**", "*.mp4"), recursive=True)
              + glob.glob(os.path.join(input_root, "**", "*.avi"), recursive=True))
    ann_files = {}
    for ann in glob.glob(os.path.join(input_root, "**", "*.txt"), recursive=True):
        ann_files[os.path.splitext(os.path.basename(ann))[0]] = ann

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    manifest = []

    for vpath in sorted(videos):
        vid = os.path.splitext(os.path.basename(vpath))[0]
        if vid not in ann_files:
            continue
        va = _read_affwild_ann(ann_files[vid])  # (F,2)
        cap = cv2.VideoCapture(vpath)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        n = min(len(frames), len(va))
        if n < 2:
            continue
        frames, va = frames[:n], va[:n]
        lands = _landmarks_from_frames(frames, extractor)
        if lands is None:
            continue
        for k, (s, e) in enumerate(_chunk_indices(n, clip_len, stride)):
            seg_va = va[s:e]
            valid = seg_va[(seg_va[:, 0] > -5) & (seg_va[:, 1] > -5)]  # -5 = marker invalid Aff-Wild2
            if valid.size == 0:
                continue
            v, a = float(valid[:, 0].mean()), float(valid[:, 1].mean())
            clip = lands[s:e]
            x = torch.from_numpy(np.ascontiguousarray(clip.transpose(2, 0, 1))).float()
            fname = f"{vid}_{k:03d}.pt"
            torch.save({"landmarks": x, "valence": v, "arousal": a,
                        "subject": vid, "source": "affwild2"},
                       os.path.join(clips_dir, fname))
            manifest.append({"file": os.path.join("clips", fname), "valence": v,
                             "arousal": a, "subject": vid, "source": "affwild2"})
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--layout", choices=["ckplus", "affwild2"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--clip_len", type=int, default=150)
    ap.add_argument("--stride", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    extractor = MediaPipeLandmarkExtractor(static_image_mode=(args.layout == "ckplus"))
    try:
        if args.layout == "ckplus":
            manifest = process_ckplus(args.input_root, args.output_dir,
                                      args.clip_len, args.stride, extractor)
        else:
            manifest = process_affwild2(args.input_root, args.output_dir,
                                        args.clip_len, args.stride, extractor)
    finally:
        extractor.close()

    mpath = os.path.join(args.output_dir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Selesai. {len(manifest)} klip -> {mpath}")


if __name__ == "__main__":
    main()
