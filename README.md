# Software-Architecture — MoodIn_Software

> Lapisan **deep learning, feature fusion, dan evaluasi** untuk **MoodIn: Smart Mirror untuk
> Klasifikasi Mood** — berbasis **ST-GCN + Graph-Transformer** (ekspresi wajah) dan
> **remote Photoplethysmography / STC-rPPG** (HR/HRV), sebagai *Personal Wellness Coach*.
> Bagian dari proyek GEMASTIK Divisi 9. Repo induk: [MoodIn](https://github.com/MoodInBME/MoodIn-master).

![layer](https://img.shields.io/badge/layer-ML-orange)
![python](https://img.shields.io/badge/python-3.12-blue?logo=python)
![status](https://img.shields.io/badge/status-WIP-yellow)

**Status:** 🚧 *Work in progress* (branch `main`). Perangkat keras (Jetson Orin Nano Super + kamera
IMX477) **masih dalam pengiriman**, sehingga pengembangan saat ini berjalan di **laptop**.

---

## 🆕 Perubahan pada push ini (untuk tim)

Revisi ini **mengunci alur fusi "VA-bridge"** pada cabang ST-GCN dan menambah pipeline dataset
sekunder + notebook uji Kaggle. Baca ringkasan berikut sebelum merge.

**Keputusan desain (menyelesaikan ambiguitas dual-head):**
- Cabang ST-GCN kini **meregresi (valence, arousal)**, bukan langsung argmax 5 kelas.
- Fusi terjadi di **ruang Valence–Arousal**, lalu **SATU** argmax kuadran di akhir:
  `valence_final = valence_wajah`, `arousal_final = w1·arousal_wajah + w2·arousal_fisio` (HR/HRV).
- Keputusan mood terjadi sekali via `labels.va_to_mood` → tidak ada dua argmax yang berkonflik saat fusion.

**Berkas yang DIMODIFIKASI (cabang `models/st_gcn_graph_tran/`):**

| Berkas | Perubahan |
|---|---|
| `model.py` | Tambah parameter `head_type` (`va`/`class`/`dual`, default `va`). ⚠️ `forward()` kini mengembalikan **dict** (`{'va': ...}` / `{'logits': ...}`). |
| `labels.py` | **Additive** (API lama utuh): tambah `EMOTION_TO_VA`, `emotion_to_va()`, `va_to_mood_id()`. `va_to_mood()` tetap mengembalikan `str`. |
| `config.yaml` | Tambah `head_type`, `transformer_ff`, `neutral_radius`, blok `loss`. |
| `train.py` | Dari kerangka → trainer penuh (dataset landmark, loss CCC, evaluasi CCC/F1, checkpoint). |

**Berkas BARU:** `models/st_gcn_graph_tran/heads.py`, `models/st_gcn_graph_tran/losses.py`,
`data/loaders/label_adapters.py`, `data/loaders/expression_video_loader.py`,
`data/preprocessing/extract_landmarks_video.py`, `evaluation/va_metrics.py`,
`notebooks/ck_plus_stgcn_kaggle.ipynb`, `docs/ST-GCN-PLAN-v2.md`.

**Tidak diubah** (dipakai apa adanya dari versi kalian): `graph.py`, `st_gcn_block.py`, `transformer.py`.

> ⚠️ **Breaking change:** output model berubah dari tensor logits menjadi **dict**. Kode lama
> `out = model(x); out.argmax(1)` harus jadi `out['va']` (head `va`, bentuk `(B, 2)`) atau
> `out['logits']` (head `class`, bentuk `(B, 5)`).

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

Alur **VA-bridge** (fusi di ruang Valence–Arousal, satu argmax di akhir):

```text
video wajah
  ├── ST-GCN + Graph-Transformer  → (valence, arousal_wajah)
  └── STC-rPPG                     → HR/HRV → arousal_fisio
fusion (late, di ruang V–A):
  valence_final = valence_wajah
  arousal_final = w1·arousal_wajah + w2·arousal_fisio
→ labels.va_to_mood(valence_final, arousal_final)   # SATU argmax kuadran → 5 mood
```

rPPG hanya menambah bukti di **sumbu arousal** (justifikasi fisiologis); keputusan mood terjadi
sekali sehingga tidak ada konflik antar-arsitektur. Bobot `w1, w2` dikalibrasi pada dataset
ber-video+fisiologi+emosi (MAHNOB-HCI / UBFC-Phys), opsional untuk laporan.

| Model | Fungsi | Lokasi | Referensi (verifikasi sitasi) |
|---|---|---|---|
| **STC-rPPG** | Estimasi HR/HRV dari video wajah → arousal fisiologis | `models/stc_rppg/` | Lin dkk. (2025) |
| **ST-GCN + Graph-Tran** | Regresi Valence–Arousal dari dinamika landmark | `models/st_gcn_graph_tran/` | Zhao dkk. (2023) |
| **Late Fusion (VA)** | Gabung arousal di ruang V–A → 1 argmax kuadran | `models/fusion/` | Antoniadis dkk. (2021) |

> **Konsistensi arsitektur:** ambiguitas *dual-head* sudah **diputuskan → VA-bridge** (regresi VA di
> ST-GCN, klasifikasi mood sekali di modul fusi). Samakan Gambar 1/4 & Bab 2.5 proposal dengan alur ini.

### 5 Kelas Mood (kuadran Valence–Arousal)

| Kelas | Kuadran | Valence | Arousal |
|---|---|---|---|
| Ceria | Q1 | + | + |
| Cemas | Q2 | − | + |
| Tertekan | Q3 | − | − |
| Rileks | Q4 | + | − |
| Netral | pusat | ~0 | ~0 |

> Zona Netral = radius `neutral_radius` (default 0.2) di sekitar pusat (0,0). Lihat `labels.py`.

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
| `data/preprocessing/extract_landmarks_video.py` | Ekspresi | Ekstraksi + cache landmark video → tensor `(C,T,V)` + manifest |
| `data/preprocessing/clahe_roi.py` | Bersama | Normalisasi pencahayaan ROI (cahaya rendah / two-way mirror) |
| `models/stc_rppg/postprocessing.py` | Khusus rPPG | Sinyal rPPG → BPM (bandpass + estimasi frekuensi) |
| `evaluation/va_metrics.py` | Evaluasi | CCC valence/arousal, akurasi kuadran mood, macro-F1 |
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
│   │   ├── prepare_data_ubfc.py       # crop wajah + simpan video & PPG ke .pt
│   │   ├── label_adapters.py          # [BARU] label per-dataset (CK+/Oulu/CAER/Aff-Wild2) → VA
│   │   └── expression_video_loader.py # [BARU] dataset landmark ter-cache → (C,T,V) + target VA
│   └── preprocessing/
│       ├── __init__.py                # ekspor subset_to_ibug68
│       ├── clahe_roi.py               # normalisasi pencahayaan ROI
│       ├── landmark_extraction_mediapipe.py
│       ├── landmark_map.py            # peta MediaPipe 468 → iBUG 68
│       ├── extract_landmarks_video.py # [BARU] ekstraksi+cache landmark video (ckplus/emotion/affwild2)
│       └── preprocessing_mediapipe.py # (lihat catatan konsolidasi di bawah)
├── models/
│   ├── __init__.py
│   ├── stc_rppg/                      # cabang rPPG (HR/HRV → arousal fisiologis)
│   │   ├── __init__.py  config.yaml  checkpoints/
│   │   ├── model.py                   # perakitan STCrPPG (kelas rakitan kanonik)
│   │   ├── ditac.py  fsfe.py  stfe.py  predictor.py  loss.py
│   │   ├── postprocessing.py          # sinyal rPPG → BPM
│   │   ├── stc_rppg.py                # (lihat catatan konsolidasi di bawah)
│   │   └── train.py                   # loop latih khusus rPPG
│   ├── st_gcn_graph_tran/             # cabang ekspresi (regresi Valence–Arousal)
│   │   ├── __init__.py  config.yaml  README.md  checkpoints/
│   │   ├── graph.py  st_gcn_block.py  transformer.py   # tidak diubah (versi tim)
│   │   ├── model.py                   # head_type va/class/dual; forward() → dict
│   │   ├── heads.py                   # [BARU] VAHead / ClassHead / DualHead
│   │   ├── losses.py                  # [BARU] CCCLoss (+ DualLoss)
│   │   ├── labels.py                  # harmonisasi VA/kategorikal; +EMOTION_TO_VA, va_to_mood_id
│   │   └── train.py                   # trainer penuh (regresi VA, loss CCC, --sanity)
│   └── fusion/
│       ├── __init__.py  dual_head_mlp.py  late_fusion.py
├── evaluation/
│   ├── metrics.py
│   ├── va_metrics.py                  # [BARU] CCC + akurasi kuadran mood + macro-F1
│   ├── confusion_matrix/
│   └── results/                       # stc_rppg_results.json  st_gcn_results.json  fusion_results.json
├── deployment/
│   ├── export_onnx.py                 # PyTorch → ONNX
│   ├── quantize_int8.py              # kuantisasi INT8 (jalur Jetson/TensorRT)
│   ├── edge_inference_rpi5.py        # inferensi Raspberry Pi 5
│   └── benchmark_latency.md
├── docs/
│   ├── CODE_REVIEW_data_layer.md  CODE_REVIEW_root_files.md  CODE_REVIEW_stc_rppg.md
│   └── CONSOLIDATION.md  MIGRATION.md  ST-GCN-PLAN.md  ST-GCN-PLAN-v2.md   # v2 [BARU] = desain VA-bridge
├── notebooks/
│   ├── 01_eda_landmarks.ipynb  02_stc_rppg_training.ipynb  03_fusion_experiments.ipynb
│   └── ck_plus_stgcn_kaggle.ipynb     # [BARU] pipeline uji ST-GCN VA di Kaggle (GPU gratis)
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

Catat lisensi & cara akses tiap dataset di `data/dataset_cards/`. Ruang label berbeda
(kategorikal vs valence–arousal) → ikuti pemetaan di `data/dataset_cards/label-harmonization.md`
dan `labels.EMOTION_TO_VA`.

| Cabang | Dataset | Peran |
|---|---|---|
| ST-GCN (ekspresi) | **Aff-Wild2** | **Utama** — in-the-wild, sudah ber-VA (butuh EULA/permintaan akses) |
| ST-GCN (ekspresi) | **CK+**, **Oulu-CASIA** | Baseline posed — cepat untuk uji pipeline |
| ST-GCN (pretrain) | AffectNet | Gambar statis — pretrain fitur per-frame (opsional) |
| STC-rPPG | **UBFC-rPPG**, PURE | Estimasi HR/HRV |

> ⚠️ **Keamanan kredensial:** jangan pernah menaruh API token (mis. Kaggle) di dalam kode.
> Pakai `~/.kaggle/kaggle.json` (chmod 600) atau env `KAGGLE_USERNAME`/`KAGGLE_KEY`, dan pastikan
> `.gitignore` memuat `kaggle.json`, `dataset/`, dan `*.pt`. Token yang pernah bocor di riwayat
> repo **harus dicabut/di-rotate**.

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

# 4. Cabang rPPG (orkestrator memilih model via argumen)
python train.py --model stc_rppg --config config/default.yaml

# 5. Cabang ekspresi ST-GCN (VA-bridge)
python -m models.st_gcn_graph_tran.train --sanity            # cek cepat arsitektur
python data/preprocessing/extract_landmarks_video.py \       # ekstraksi+cache landmark (SEKALI)
    --input_root dataset/ckplus --layout ckplus \
    --output_dir data/prepared/ckplus --clip_len 150
python -m models.st_gcn_graph_tran.train \                   # latih regresi VA (loss CCC)
    --config models/st_gcn_graph_tran/config.yaml \
    --manifest data/prepared/ckplus/manifest.json --val_split 0.2

# 6. Inferensi satu video (butuh bobot terlatih untuk BPM & mood valid)
python inference.py
```

> Untuk **Aff-Wild2** (setelah akses EULA): pakai `--layout affwild2` pada langkah 5 (klip video +
> file anotasi VA `.txt`). Testing GPU gratis: lihat `notebooks/ck_plus_stgcn_kaggle.ipynb`.

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

- [x] Kunci alur fusi **VA-bridge** (regresi VA di ST-GCN, 1 argmax di fusi).
- [ ] Konsolidasi 3 pasang berkas duplikat (lihat `docs/CONSOLIDATION.md`).
- [ ] Putuskan board edge (Jetson vs Pi 5) lalu selaraskan `deployment/` + README.
- [ ] Verifikasi visual peta `landmark_map.py` (overlay 68 titik ke wajah sampel).
- [ ] Ajukan akses **Aff-Wild2** (EULA) → ganti dataset dari baseline CK+.
- [ ] Selaraskan `CATEGORICAL_MAP` (labels.py) vs koordinat `EMOTION_TO_VA` (lihat catatan di bawah).
- [ ] Isi `data/preprocessing/clahe_roi.py` bila masih kosong.
- [ ] Lengkapi `train.py` (orkestrator) + trainer per-model.
- [ ] Git LFS/DVC untuk `checkpoints/*.pth` & bobot ONNX.
- [ ] Verifikasi sitasi (Lin 2025, Zhao 2023, Antoniadis 2021).

> **Catatan harmonisasi label:** `labels.CATEGORICAL_MAP` (mis. `surprise → Ceria`, `disgust →
> Tertekan`) dan koordinat `labels.EMOTION_TO_VA` bisa berbeda untuk emosi ambigu. Keduanya
> dipertahankan; tetapkan satu acuan sebelum training Aff-Wild2.

## 📚 Status Eksperimen

Lihat `evaluation/results/` (`stc_rppg_results.json`, `st_gcn_results.json`, `fusion_results.json`).
