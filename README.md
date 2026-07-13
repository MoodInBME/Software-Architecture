# Software-Architecture

Arsitektur deep learning, feature fusion, dan evaluasi model MoodIn.
Bagian dari proyek [MoodIn](https://github.com/MoodInBME/MoodIn-master.git).

![Repo](https://img.shields.io/badge/github-repo-blue?logo=github)
![Layer](https://img.shields.io/badge/layer-ML-orange)

## Pipeline Model

Alur: video wajah → (a) ST-GCN/Graph-Tran mengekstrak intensitas Action Unit; (b) STC-rPPG mengestimasi HR/HRV → fitur (a)+(b) difusikan → dual-head menghasilkan regresi valence–arousal + klasifikasi 5 kelas.

| Model | Fungsi | Referensi |
| --- | --- | --- |
| STC-rPPG | Estimasi HR/HRV dari video wajah | Lin dkk. (2025) — _verifikasi sitasi_ |
| ST-GCN + Graph-Tran | Ekstraksi intensitas Action Unit | Zhao dkk. (2023) — _verifikasi sitasi_ |
| Dual-Head Fusion | Regresi V–A + klasifikasi 5 kelas | Antoniadis dkk. (2021) — _verifikasi sitasi_ |

> Konsistensi arsitektur: pipeline di atas (ST-GCN → AU → fusi → dual-head) harus sama persis dengan Gambar 1/4 dan Bab 2.5 proposal. Pada draf proposal ada ambiguitas apakah dual-head melekat pada ST-GCN atau pada modul fusi — samakan satu versi di kedua tempat sebelum evaluasi.

## 5 Kelas Mood

Ceria (Q1), Cemas (Q2), Tertekan (Q3), Rileks (Q4), Netral (pusat) — memetakan kuadran valence–arousal (Gambar 2 proposal).

## Dataset

Catat lisensi & cara akses tiap dataset di `data/dataset_cards/`. Karena dataset ekspresi (mis. CK+, AffectNet) dan dataset rPPG (PURE, UBFC-rPPG) punya ruang label berbeda (kategorikal vs valence–arousal), sertakan dokumen harmonisasi label → kuadran V–A agar pelatihan gabungan valid. Jika proposal menyebut dataset lain (Oulu-CASIA, CAER, Aff-Wild2), samakan daftarnya di sini.

## Setup

Gunakan **satu** manajer environment sebagai kanonik agar hasil reprodusibel. `environment.yml` (conda) dijadikan sumber utama; `requirements.txt` diturunkan darinya.

```bash
conda env create -f environment.yml
conda activate moodin-ml
# requirements.txt hanya untuk lingkungan non-conda
```

Target deployment edge: Raspberry Pi 5 (lihat `deployment/edge_inference_rpi5.py`). Pin versi Python/PyTorch di `environment.yml`; hindari versi yang belum stabil di Debian Trixie.

## Struktur Repository

```text
Software-Architecture/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitattributes              # untuk Git LFS
├── .gitignore
├── config/
│   └── default.yaml            # hyperparameter global
├── data/
│   ├── preprocessing/
│   │   ├── clahe_roi.py
│   │   └── landmark_extraction_mediapipe.py
│   ├── dataset_cards/
│   │   ├── ck-plus.md
│   │   ├── affectnet.md
│   │   ├── pure.md
│   │   ├── ubfc-rppg.md          # catat lisensi & cara akses tiap dataset
│   │   └── label-harmonization.md   # pemetaan label → kuadran V–A
│   └── loaders/
│       └── dataset_loader.py
├── models/
│   ├── stc_rppg/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── config.yaml
│   │   └── checkpoints/          # dikelola via Git LFS/DVC
│   ├── st_gcn_graph_tran/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── checkpoints/
│   └── fusion/
│       ├── late_fusion.py
│       └── dual_head_mlp.py      # head regresi VA + klasifikasi 5 kelas
├── evaluation/
│   ├── metrics.py                # implementasi Pers. 11–13
│   ├── confusion_matrix/
│   └── results/
│       ├── stc_rppg_results.json
│       ├── st_gcn_results.json
│       └── fusion_results.json
├── deployment/
│   ├── export_onnx.py
│   ├── quantize_int8.py
│   ├── edge_inference_rpi5.py
│   ├── mqtt_publisher.py         # publish hasil ke moodin/inference/result
│   └── benchmark_latency.md
├── notebooks/
│   ├── 01_eda_landmarks.ipynb
│   ├── 02_stc_rppg_training.ipynb
│   └── 03_fusion_experiments.ipynb
└── tests/
    ├── test_preprocessing.py
    └── test_models.py
```

## Integrasi (MQTT)

Hasil inferensi dipublikasikan ke topik `moodin/inference/result` sesuai kontrak kanonik di `MoodIn-master/docs/ipc-contract.md`. Modul `deployment/mqtt_publisher.py` yang bertanggung jawab; UI (`Interface-Display`) hanya mengonsumsi, tidak meng-import kode repo ini.

## Status Eksperimen

Lihat [evaluation/results/](evaluation/results/).
