"""Inferensi STC-rPPG (versi diperbaiki - lihat S1/S4 di CODE_REVIEW_root_files.md).

Perubahan dari versi asli:
- Import via paket (`from models import STCrPPG`) supaya kebal terhadap tabrakan
  nama modul/paket `stc_rppg` (S1).
- `torch.load(..., weights_only=True)` untuk keamanan & kompat torch >= 2.6 (S4).
- Pakai FaceMeshVideoProcessor (MediaPipe) yang mengembalikan (C,T,H,W); batch
  ditambahkan di sini dengan unsqueeze(0).
"""
import os
import torch

from models import STCrPPG
from postprocessing import rPPGProcessor

try:
    # Disarankan (S2): pipeline MediaPipe tunggal untuk ROI + landmark.
    from suggested.preprocessing_mediapipe import FaceMeshVideoProcessor as VideoProcessor
except Exception:  # fallback ke versi lama bila MediaPipe belum dipasang
    from preprocessing import FaceVideoProcessor as VideoProcessor


def run_inference(video_path, weight_path=None):
    print("=== Memulai Inferensi STC-rPPG ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device komputasi: {device}")

    processor = VideoProcessor(target_size=(128, 128), seq_length=128)
    model = STCrPPG().to(device)
    post = rPPGProcessor(fps=30.0)

    if weight_path and os.path.exists(weight_path):
        print(f"Memuat bobot: {weight_path}")
        state = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        print("Peringatan: bobot tidak ditemukan. Output BPM tidak valid (init acak).")

    model.eval()

    try:
        clip = processor.process_video(video_path)
        if clip is None:
            print("Gagal memproses video (wajah tidak terdeteksi).")
            return None
        # (C,T,H,W) -> (1,C,T,H,W)
        if clip.dim() == 4:
            clip = clip.unsqueeze(0)
        clip = clip.to(device)

        with torch.no_grad():
            signal = model(clip)
        raw = signal.squeeze().cpu().numpy()

        clean = post.filter_signal(raw)
        bpm = post.calculate_bpm(clean)
        print("\n" + "=" * 40)
        print(f" ESTIMASI DETAK JANTUNG: {bpm:.2f} BPM")
        print("=" * 40 + "\n")
        return bpm
    except Exception as e:
        print(f"Kesalahan saat inferensi: {e}")
        return None


if __name__ == "__main__":
    run_inference("sample.mp4", "stc_rppg_weights.pth")
