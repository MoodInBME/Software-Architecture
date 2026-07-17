import torch
import numpy as np
import os

# Mengimpor modul-modul yang telah kita bangun
from preprocessing import FaceVideoProcessor
from models.stc_rppg import STCrPPG
from postprocessing import rPPGProcessor

def run_inference(video_path, weight_path=None):
    print("=== Memulai Inferensi STC-rPPG ===")
    
    # 1. Konfigurasi Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device komputasi: {device}")

    # 2. Inisialisasi Modul
    print("\nMempersiapkan pipeline...")
    video_processor = FaceVideoProcessor(target_size=(128, 128), seq_length=128)
    model = STCrPPG().to(device)
    post_processor = rPPGProcessor(fps=30.0)

    # 3. Memuat Bobot Model (Jika sudah dilatih)
    if weight_path and os.path.exists(weight_path):
        print(f"Memuat bobot model dari {weight_path}...")
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print("Peringatan: Bobot model tidak ditemukan. Menggunakan inisialisasi acak (hasil BPM tidak valid).")

    # Set model ke mode evaluasi (mematikan dropout dan batch norm updates)
    model.eval()

    try:
        # 4. Pra-pemrosesan Data
        print(f"\nMemproses video '{video_path}'...")
        # Output video_tensor memiliki shape: (1, 3, 128, 128, 128)
        video_tensor = video_processor.process_video(video_path)
        
        if video_tensor is None:
            print("Gagal memproses video. Pastikan wajah terlihat jelas.")
            return

        # Pindahkan tensor ke GPU/CPU
        video_tensor = video_tensor.to(device)

        # 5. Eksekusi Model (Forward Pass)
        print("Mengekstrak sinyal fisiologis melalui jaringan saraf...")
        with torch.no_grad(): # Mematikan perhitungan gradien agar memori lebih hemat
            predicted_signal = model(video_tensor)
            
        # Pindahkan sinyal kembali ke CPU dan konversi ke numpy array 1D
        raw_rppg = predicted_signal.squeeze().cpu().numpy()

        # 6. Pasca-pemrosesan Data
        print("Menyaring sinyal dan mengestimasi Detak Jantung (BPM)...")
        clean_rppg = post_processor.filter_signal(raw_rppg)
        bpm = post_processor.calculate_bpm(clean_rppg)

        print("\n" + "="*40)
        print(f" HASIL ESTIMASI DETAK JANTUNG: {bpm:.2f} BPM")
        print("="*40 + "\n")

    except Exception as e:
        print(f"Terjadi kesalahan saat inferensi: {e}")

if __name__ == "__main__":
    # Tentukan nama file video uji (dummy data) Anda
    target_video = "sample.mp4"
    
    # Path ke file bobot model jika Anda sudah menyimpannya melalui train.py
    # Ganti menjadi None jika hanya ingin menguji aliran program
    target_weights = "stc_rppg_weights.pth" 
    
    run_inference(target_video, target_weights)