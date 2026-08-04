# Software-Architecture

Seluruh kode model dan pemrosesan sinyal MoodIn. Repo ini tidak memuat kode UI dan tidak memuat kode driver perangkat keras.

![layer](https://img.shields.io/badge/layer-ML-orange)
![python](https://img.shields.io/badge/python-3.12-blue?logo=python)
![status](https://img.shields.io/badge/status-WIP-yellow)

---

## 1. Struktur Repo

```
Software-Architecture/
  models/
    st_gcn_graph_tran/           cabang ekspresi (ST-GCN + Graph Transformer)
      model.py, graph.py, heads.py, st_gcn_block.py, transformer.py
      train.py, config.yaml, labels.py
    stc_rppg/                    cabang fisiologis (STC-rPPG)
      model.py, fsfe.py, stfe.py, predictor.py, cdc.py, ditac.py
    fusion/
      late_fusion.py             LateFusionVABridge
  data/
    preprocessing/
      extract_landmarks_video.py MediaPipe 468 -> iBUG 68 -> cache .pt + manifest
      landmark_map.py            MEDIAPIPE_FACEMESH_TO_IBUG68 (68 indeks)
      landmark_extraction_mediapipe.py
    loaders/
      expression_video_loader.py ExpressionLandmarkDataset -> (x, va, mood)
  evaluation/
    va_metrics.py                CCC, PCC, RMSE, SAGR, evaluate_va
  notebooks/
    ck_plus_stgcn_kaggle.ipynb   pelatihan + validasi + ekspor + mock UI
  weights/                       bobot model (dummy atau terlatih)
  inference.py                   pipeline real-time edge (Jetson/laptop)
  train.py                       router pelatihan dua cabang
  JETSON_SETUP.md                panduan setup Jetson Orin Nano Super
```

## 2. Dua Cabang, Satu Ruang Keluaran

| Parameter | Cabang Ekspresi | Cabang Fisiologis |
|---|---|---|
| Masukan | `(2, 150, 68)` landmark ternormalisasi | `(3, 128, H, W)` patch ROI |
| Model | `STGCN_GraphTransformer` | `STCrPPG` |
| Keluaran | valence, arousal_wajah | sinyal BVP -> HR, HRV, RR -> arousal_fisio |
| Parameter | ~0,82 M | ~0,82 M |
| Mode Gagal | wajah tidak terdeteksi, oklusi | cahaya rendah, gerakan, kulit gelap |

Keduanya bertemu di `LateFusionVABridge`, yang memetakan arousal fisiologis ke sumbu yang sama dengan arousal ekspresi lalu menggabungkannya dengan bobot `w_face_arousal` dan `w_physio_arousal`, ditambah EMA untuk meredam kedip.

Kalau kanal fisiologis timeout (misal cahaya terlalu rendah), fusi akan mengandalkan fitur ekspresi saja dan mensinyalkan `physio_used=false`. Ini bukan kegagalan, ini perilaku degradasi terencana yang wajib direspons oleh UI cermin.

## 3. Pipeline Kaggle Notebook

`notebooks/ck_plus_stgcn_kaggle.ipynb` adalah alur ujung-ke-ujung (end-to-end) model (gratis dengan GPU Kaggle T4). 

* **Split subject-independent:** Memakai `StratifiedGroupKFold`. Asersi ketat memastikan subjek yang sama tidak akan bocor lintas-fold, guna mencegah model hanya belajar "mengenali wajah" daripada "mengenali ekspresi".
* **Validasi Kategorikal & Dimensional:** Mendukung matriks Macro-F1 (menghindari bias kelas mayoritas) serta CCC (Concordance Correlation Coefficient) terpisah untuk Valence dan Arousal.
* **Uji Falsifikasi (Baseline):** Termasuk pembanding SVM/MLP/Dummy di atas landmark pipih (flattened) untuk menjustifikasi urgensi pemakaian GCN (Graph Convolutional).
* **Mock Publisher:** Ekspor otomatis `mock_inference_publisher.py` untuk mengizinkan tim UI mengembangkan cermin secara paralel walau model belum selesai dilatih penuh.

## 4. Lingkungan Pengembangan

```text
Python 3.12 (Native PyTorch 2.3.1 + torchvision 0.18.1)
numpy>=1.24,<2.0
mediapipe==0.10.14
opencv-python==4.10.0.84
paho-mqtt==2.1.0
scikit-learn  onnx  onnxruntime  matplotlib  pyyaml
```
> **Catatan Edge:** Deployment di Jetson Orin Nano Super menggunakan JetPack 7.2 (CUDA 13.2.1). PyTorch: `pip install torch==2.12.0+cu132 --index-url https://download.pytorch.org/whl/cu132`

## 5. Batasan Terbuka (Catatan Evaluasi)

- CK+ adalah ekspresi **berakting** di laboratorium, bukan ekspresi spontan.
- Label VA diturunkan dari kategori emosi diskrit, bukan anotasi VA kontinu asli. Aff-Wild2 diperlukan untuk validasi VA sejati (EULA sedang diproses).
- Belum ada validasi akurasi lintas warna kulit (skala Fitzpatrick) serta uji transmisi cahaya cermin secara empiris di bawah 30%.
- Kinerja FPS harus dievaluasi menggunakan Jetson, angka di notebook Kaggle T4 bukan rujukan final perangkat keras.

## 6. Tugas Tersisa

1. Tambahkan argumen `--folds` dan `--fold` ke `models/st_gcn_graph_tran/train.py`.
2. Verifikasi empiris `MEDIAPIPE_TO_IBUG68` (Procrustes) terhadap LCK+.
3. Kalibrasi `w1`, `w2` fusi multimodal dengan metrik `oof_predictions.npz`.
