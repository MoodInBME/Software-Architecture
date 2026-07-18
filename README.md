# Software-Architecture — MoodIn_Software

> Lapisan **deep learning, feature fusion, dan evaluasi** untuk **MoodIn: Smart Mirror untuk
> Klasifikasi Mood** — berbasis **ST-GCN + Graph-Transformer** (ekspresi/Action Unit) dan
> **remote Photoplethysmography / STC-rPPG** (HR/HRV), sebagai *Personal Wellness Coach*.
> Bagian dari proyek GEMASTIK Divisi 9. Repo induk: [MoodIn](https://github.com/MoodInBME/MoodIn-master).

![layer](https://img.shields.io/badge/layer-ML-orange)
![python](https://img.shields.io/badge/python-3.12-blue?logo=python)
![status](https://img.shields.io/badge/status-WIP-yellow)

**Status:** 🚧 *Work in progress* (branch `main`). Perangkat keras (Jetson Orin Nano Super + kamera
IMX477) **masih dalam pengiriman**, sehingga pengembangan saat ini berjalan di **laptop**.

---

## ⚙️ Environment yang Disarankan

| Komponen | Pilihan | Catatan |
|---|---|---|
| **Python** | **3.12** | Didukung penuh PyTorch 2.3.x & MediaPipe 0.10.x. **Jangan pakai versi lawas** (3.8/3.9) maupun **3.14** (wheel belum tersedia). |
| **PyTorch** | **2.3.1** + torchvision 0.18.1 | Laptop via pip/conda. Board edge memakai wheel khusus (lihat *Deployment*). |
| **NumPy** | `>=1.24,<2.0` | Kompatibilitas MediaPipe/OpenCV saat ini. |
| **Deteksi wajah** | **MediaPipe FaceMesh** | Satu detektor untuk ROI rPPG **dan** landmark ST-GCN (menggantikan dlib). |
| **Manajer env** | conda/mamba (kanonik) | `environment.yml` = sumber utama; `requirements.txt` diturunkan darinya. |

### Setup (laptop / dev)

```bash
# Opsi A — conda/mamba (kanonik, reprodusibel)
conda env create -f environment.yml
conda activate moodin-ml

# Opsi B — venv + pip (lingkungan non-conda)
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Catatan board edge:** JetPack 6.2 (Jetson) membawa Python 3.10 secara *native*. Dev di laptop
> tetap **3.12**; untuk deployment gunakan `requirements-jetson.txt` + wheel resmi NVIDIA (jp6),
> **bukan** PyPI, atau container/conda yang disesuaikan ke 3.10 di perangkat.

---

## 🧠 Pipeline Model

Alur: video wajah → **(a)** ST-GCN/Graph-Transformer mengekstrak fitur ekspresi (intensitas
Action Unit); **(b)** STC-rPPG mengestimasi HR/HRV → fitur (a)+(b) **difusikan** → *dual-head*
menghasilkan **regresi valence–arousal + klasifikasi 5 kelas mood**.

| Model | Fungsi | Lokasi | Referensi (verifikasi sitasi) |
|---|---|---|---|
| **STC-rPPG** | Estimasi HR/HRV dari video wajah | `models/stc_rppg/` | Lin dkk. (2025) |
| **ST-GCN + Graph-Tran** | Fitur ekspresi / Action Unit | `models/st_gcn_graph_tran/` | Zhao dkk. (2023) |
| **Dual-Head Fusion** | Regresi V–A + klasifikasi 5 kelas | `models/fusion/` | Antoniadis dkk. (2021) |

> **Konsistensi arsitektur:** samakan pipeline ini dengan Gambar 1/4 dan Bab 2.5 proposal.
> Ada ambiguitas apakah *dual-head* melekat pada ST-GCN atau modul fusi — tetapkan **satu** versi
> di README, kode, dan proposal sebelum evaluasi.

### 5 Kelas Mood (kuadran Valence–Arousal)

| Kelas | Kuadran | Valence | Arousal |
|---|---|---|---|
| Ceria | Q1 | + | + |
| Cemas | Q2 | − | + |
| Tertekan | Q3 | − | − |
| Rileks | Q4 | + | − |
| Netral | pusat | ~0 | ~0 |

---

## 🧭 Lapisan Kode & Tanggung Jawab

Tiga lapisan agar tiap skrip punya rumah yang jelas:

1. **Infrastruktur data (bersama)** — dipakai SEMUA model → `data/`
2. **Spesifik per model** — hanya milik satu cabang → `models/<nama>/`
3. **Orkestrasi sistem** — lintas model → root (`train.py`, `inference.py`)

| Berkas | Lapisan | Catatan |
|---|---|---|
| `data/preprocessing/landmark_extraction_mediapipe.py` | Bersama | Ekstraksi wajah + landmark (dipakai rPPG & ST-GCN) |
| `data/preprocessing/landmark_map.py` | Bersama | Peta **MediaPipe 468 → iBUG 68** (`subset_to_ibug68`) |
| `data/preprocessing/clahe_roi.py` | Bersama | Normalisasi pencahayaan ROI (cahaya rendah / two-way mirror) |
| `models/stc_rppg/postprocessing.py` | Khusus rPPG | Sinyal rPPG → BPM (bandpass + estimasi frekuensi) |
| `tests/` | Test | `main.py` (smoke), `test_ditac.py`, `test_models.py`, `test_preprocessing.py` |
| `train.py`, `inference.py` (root) | Orkestrasi | Dispatch lintas model |

> **Jembatan preprocessing → ST-GCN:** model memakai `num_node: 68`, sedangkan FaceMesh
> mengeluarkan 468 titik. Selalu lewati `subset_to_ibug68()` sebelum menyusun tensor
> `(N, C, T, V=68)`. Urutan 68 titik mengikuti skema iBUG dan **harus** cocok dengan `graph.py`.

---

## 📁 Struktur Repository

```text
MoodIn_Software/
├── config/
│   └── default.yaml                    # hyperparameter & path global
├── data/
│   ├── __init__.py
│   ├── dataset_cards/                 # afektif: affectnet, ck-plus, pure; rppg: ubfc-rppg
│   │   ├── affectnet.md  ck-plus.md  pure.md  ubfc-rppg.md
│   │   └── label-harmonization.md     # pemetaan label → kuadran V-A / 5 kelas
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py          # TensorClipDataset generik (.pt)
│   │   ├── dataset_ubfc_loader.py     # UBFCrPPGDataset (jendela geser + PPG tersinkron)
│   │   ├── download_ubfc_kaggle.py    # unduh dataset (kredensial via env/kaggle.json)
│   │   └── prepare_data_ubfc.py       # crop wajah + simpan video & PPG ke .pt
│   └── preprocessing/
│       ├── __init__.py                # ekspor subset_to_ibug68
│       ├── clahe_roi.py               # normalisasi pencahayaan ROI
│       ├── landmark_extraction_mediapipe.py
│       ├── landmark_map.py            # peta MediaPipe 468 → iBUG 68
│       └── preprocessing_mediapipe.py # (lihat catatan konsolidasi di bawah)
├── models/
│   ├── __init__.py
│   ├── stc_rppg/                      # cabang rPPG (HR/HRV)
│   │   ├── __init__.py  config.yaml  checkpoints/
│   │   ├── model.py                   # perakitan STCrPPG (kelas rakitan kanonik)
│   │   ├── ditac.py  fsfe.py  stfe.py  predictor.py  loss.py
│   │   ├── postprocessing.py          # sinyal rPPG → BPM
│   │   ├── stc_rppg.py                # (lihat catatan konsolidasi di bawah)
│   │   └── train.py                   # loop latih khusus rPPG
│   ├── st_gcn_graph_tran/             # cabang ekspresi (Action Unit → 5 kelas)
│   │   ├── __init__.py  config.yaml  README.md  checkpoints/
│   │   ├── graph.py  st_gcn_block.py  transformer.py  model.py
│   │   ├── labels.py                  # harmonisasi VA/kategorikal → 5 kelas
│   │   └── train.py                   # kerangka latih (mode --sanity)
│   └── fusion/
│       ├── __init__.py  dual_head_mlp.py  late_fusion.py
├── evaluation/
│   ├── metrics.py
│   ├── confusion_matrix/
│   └── results/                       # stc_rppg_results.json  st_gcn_results.json  fusion_results.json
├── deployment/
│   ├── export_onnx.py                 # PyTorch → ONNX
│   ├── quantize_int8.py              # kuantisasi INT8 (jalur Jetson/TensorRT)
│   ├── edge_inference_rpi5.py        # inferensi Raspberry Pi 5
│   └── benchmark_latency.md
├── docs/
│   ├── CODE_REVIEW_data_layer.md  CODE_REVIEW_root_files.md  CODE_REVIEW_stc_rppg.md
│   └── CONSOLIDATION.md  MIGRATION.md  ST-GCN-PLAN.md
├── notebooks/
│   ├── 01_eda_landmarks.ipynb  02_stc_rppg_training.ipynb  03_fusion_experiments.ipynb
├── tests/
│   ├── __init__.py  main.py  test_ditac.py  test_models.py  test_preprocessing.py
├── .gitattributes  .gitignore
├── environment.yml  requirements.txt  requirements-jetson.txt
├── inference.py                       # inferensi end-to-end (orkestrator)
└── train.py                           # orkestrator training (dispatch per-model)
```

> **Catatan konsolidasi (lihat `docs/CONSOLIDATION.md`):** masih ada tiga pasang berkas yang
> perlu digabung/dibersihkan sebelum rilis — `models/stc_rppg/{model.py, stc_rppg.py}`,
> `tests/{main.py, test_models.py}`, dan `data/preprocessing/{landmark_extraction_mediapipe.py,
> preprocessing_mediapipe.py}`. Tetapkan satu berkas kanonik per pasangan.

---

## 💾 Dataset

Catat lisensi & cara akses tiap dataset di `data/dataset_cards/`. Karena dataset ekspresi
(CK+, AffectNet) dan rPPG (PURE, UBFC-rPPG) punya ruang label berbeda (kategorikal vs
valence–arousal), ikuti pemetaan di `data/dataset_cards/label-harmonization.md`.

> ⚠️ **Keamanan kredensial:** jangan pernah menaruh API token (mis. Kaggle) di dalam kode.
> Pakai `~/.kaggle/kaggle.json` (chmod 600) atau env `KAGGLE_USERNAME`/`KAGGLE_KEY`, dan pastikan
> `.gitignore` memuat `kaggle.json`, `dataset/`, dan `*.pt`.

---

## ▶️ Cara Menjalankan (dev di laptop)

```bash
# 1. Unduh dataset (kredensial via env / kaggle.json, TANPA token di kode)
python data/loaders/download_ubfc_kaggle.py --path dataset

# 2. Prapemrosesan UBFC-rPPG -> .pt per subjek (video + PPG tersinkron)
python data/loaders/prepare_data_ubfc.py \
    --dataset_dir dataset/DATASET_2 --output_dir data/prepared/ubfc

# 3. Smoke test arsitektur & unit test
pytest tests/

# 4. Latih model (orkestrator memilih model via argumen)
python train.py --model stc_rppg --config config/default.yaml
python models/st_gcn_graph_tran/train.py --sanity     # cek cepat cabang ekspresi

# 5. Inferensi satu video (butuh bobot terlatih untuk BPM valid)
python inference.py
```

---

## 🚀 Deployment (Edge)

> **Keputusan board belum final — tetapkan satu board kanonik.**

- **Jetson Orin Nano Super** (sedang dikirim): `deployment/export_onnx.py` →
  `deployment/quantize_int8.py` → (inferensi TensorRT). Pakai `requirements-jetson.txt`.
- **Raspberry Pi 5** (opsi README awal): `deployment/edge_inference_rpi5.py` + ONNX Runtime.

Catat hasil pengukuran di `deployment/benchmark_latency.md`. Setelah board diputuskan,
selaraskan bagian ini, `requirements`, dan skrip `deployment/` (mis. tambah `edge_inference_jetson.py`).

---

## 🔌 Integrasi (MQTT)

Hasil inferensi dipublikasikan ke topik `moodin/inference/result` sesuai kontrak kanonik di
`MoodIn-master/docs/ipc-contract.md`. UI (`Interface-Display`) hanya **mengonsumsi**, tidak
meng-import kode repo ini.

## 🔗 Keterkaitan Repo MoodIn

- **MoodIn** — hub/dokumentasi utama proyek
- **Hardware-Connections** — skema perangkat keras (board, kamera, daya)
- **Interface-Display** — UI cermin (PySide6 + QML), berkomunikasi via MQTT

---

## 📌 Status & TODO

- [ ] Konsolidasi 3 pasang berkas duplikat (lihat `docs/CONSOLIDATION.md`).
- [ ] Putuskan board edge (Jetson vs Pi 5) lalu selaraskan `deployment/` + README.
- [ ] Verifikasi visual peta `landmark_map.py` (overlay 68 titik ke wajah sampel).
- [ ] Isi `data/preprocessing/clahe_roi.py` bila masih kosong.
- [ ] Lengkapi `train.py` (orkestrator) + trainer per-model.
- [ ] Git LFS/DVC untuk `checkpoints/*.pth` & bobot ONNX.
- [ ] Verifikasi sitasi (Lin 2025, Zhao 2023, Antoniadis 2021).

## 📚 Status Eksperimen

Lihat `evaluation/results/` (`stc_rppg_results.json`, `st_gcn_results.json`, `fusion_results.json`).
