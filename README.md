# Software-Architecture

Arsitektur deep learning, feature fusion, dan evaluasi model MoodIn.
Bagian dari proyek [MoodIn](../../MoodIn-master).

![https://img.shields.io/badge/github-repo-blue?logo=github](https://img.shields.io/badge/github-repo-blue?logo=github)

## Model
| Model | Fungsi | Referensi |
|---|---|---|
| STC-rPPG | Estimasi HR/HRV dari video wajah | Lin dkk. (2025) |
| ST-GCN + Graph-Tran | Ekstraksi intensitas AU | Zhao dkk. (2023) |
| Dual-Head Fusion | Regresi V-A + Klasifikasi 5-kelas | Antoniadis dkk. (2021) |

## Setup
```bash
conda env create -f environment.yml
pip install -r requirements.txt
```

## Struktur Root
Lihat pohon direktori lengkap di [docs](../MoodIn-master/docs)

## Struktur Repository
```bash
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
│   │   └── ubfc-rppg.md         # catat lisensi & cara akses tiap dataset
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
│       └── dual_head_mlp.py      # head regresi VA + klasifikasi 5-kelas
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
│   └── benchmark_latency.md
├── notebooks/
│   ├── 01_eda_landmarks.ipynb
│   ├── 02_stc_rppg_training.ipynb
│   └── 03_fusion_experiments.ipynb
└── tests/
    ├── test_preprocessing.py
    └── test_models.py
```

## Status Eksperimen
Lihat [evaluation/results/](evaluation/results/)