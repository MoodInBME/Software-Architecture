"""
MoodIn - Training Orchestrator (Root)

Sesuai aturan arsitektur (Late Fusion), cabang wajah (ST-GCN) dan cabang 
fisiologis (STC-rPPG) dilatih secara TERPISAH karena berasal dari dua modalitas 
dan dataset yang berbeda (CK+/Aff-Wild2 vs UBFC-rPPG).

Setelah bobot keduanya konvergen, tahap fusi (LateFusionVABridge) hanya bertugas
menyelaraskan ruang arousal (w_face_arousal dan w_physio_arousal) di saat inferensi.

Script ini memandu pengguna untuk mengeksekusi pipeline pelatihan yang benar.
"""

def main():
    print("="*65)
    print(" MoodIn - Training Orchestrator ".center(65, "="))
    print("="*65)
    print("Mengingat arsitektur Late Fusion, cabang wajah dan fisiologis")
    print("dilatih secara independen sebelum difusi pada saat inferensi.\n")
    
    print("[1] CABANG EKSPRESI (ST-GCN Graph-Transformer)")
    print("    Sangat disarankan melatih ini di Kaggle T4 GPU (gratis).")
    print("    Jalankan notebook: notebooks/ck_plus_stgcn_kaggle.ipynb")
    print("    Atau jalankan script lokal:")
    print("    $ python -m models.st_gcn_graph_tran.train \\\n        --config models/st_gcn_graph_tran/config.yaml\n")
    
    print("[2] CABANG FISIOLOGIS (STC-rPPG)")
    print("    Jalankan script lokal (pastikan data UBFC sudah di-prep):")
    print("    $ python -m models.stc_rppg.train \\\n        --config models/stc_rppg/config.yaml\n")
    
    print("[3] INFERENSI END-TO-END (Real-time Jetson)")
    print("    Jika Anda ingin menguji integrasi fusi secara real-time,")
    print("    silakan gunakan `python inference.py` dengan bobot terlatih.")
    print("="*65)

if __name__ == "__main__":
    main()