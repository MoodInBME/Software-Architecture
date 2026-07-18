# Migrasi Tata Letak Skrip Root -> Struktur README

Keenam skrip di root `Software-Architecture/` sekarang mencampur **tiga lapisan**:

1. **Infrastruktur bersama** (dipakai SEMUA model) -> `data/preprocessing/`
2. **Khusus STC-rPPG** (hanya bermakna untuk cabang rPPG) -> `models/stc_rppg/` & `tests/`
3. **Orkestrasi sistem** (lintas model) -> tetap di root

## Pemetaan file

| File root sekarang | Isi | Sifat | Lokasi tujuan |
| --- | --- | --- | --- |
| `preprocessing.py` (`FaceVideoProcessor`) | crop wajah dari video | **Bersama** (ST-GCN butuh wajah/landmark juga) | `data/preprocessing/landmark_extraction_mediapipe.py` |
| `postprocessing.py` (`rPPGProcessor`) | sinyal rPPG -> BPM (bandpass+FFT) | **Khusus rPPG** | `models/stc_rppg/postprocessing.py` |
| `main.py` | smoke test `STCrPPG` | Test | `tests/test_models.py` |
| `test_ditac.py` | unit test DiTAC | Test | `tests/test_ditac.py` |
| `train.py` | loop latih (kerangka) | Orkestrasi + rPPG | root `train.py` (orkestrator) + `models/stc_rppg/train.py` (loop nyata) |
| `inference.py` | pipeline end-to-end | Orkestrasi (kini jalur rPPG) | root `inference.py` |

## Kenapa `postprocessing.py` TIDAK boleh di root

"Post-processing" di root menyiratkan tahap global, padahal isinya khusus rPPG (bandpass 0.7-3.5 Hz
-> BPM). ST-GCN butuh post-processing yang sama sekali berbeda (softmax -> label + penghalusan
temporal). Jadi menaruh satu `postprocessing.py` global itu menyesatkan. Pindahkan ke
`models/stc_rppg/postprocessing.py`. Kelak ST-GCN punya post-processing sendiri di paketnya.

## Kenapa `preprocessing.py` pindah ke `data/preprocessing/`

Ekstraksi/crop wajah dibutuhkan **kedua** cabang (rPPG dan ST-GCN). Itu infrastruktur data
bersama, bukan milik rPPG. Sekaligus migrasi dlib -> MediaPipe (satu detektor: ROI + landmark).

## Perintah git (jalankan di root repo)

    git mv preprocessing.py  data/preprocessing/landmark_extraction_mediapipe.py
    git mv postprocessing.py models/stc_rppg/postprocessing.py
    git mv test_ditac.py     tests/test_ditac.py
    git mv main.py           tests/test_models.py
    # train.py & inference.py tetap di root (lihat penyesuaian import di bawah)

## Penyesuaian import (WAJIB setelah pindah)

**`inference.py` (root):**

    # lama
    from preprocessing import FaceVideoProcessor
    from models.stc_rppg import STCrPPG
    from postprocessing import rPPGProcessor
    # baru
    from data.preprocessing.landmark_extraction_mediapipe import FaceMeshVideoProcessor
    from models.stc_rppg import STCrPPG                 # via models/stc_rppg/__init__.py
    from models.stc_rppg.postprocessing import rPPGProcessor

**`tests/test_ditac.py`:**

    # lama:  from models.ditac import DiTAC
    # baru:  from models.stc_rppg.ditac import DiTAC

**`tests/test_models.py` (eks-main.py):**

    from models.stc_rppg import STCrPPG   # tetap, komponen kini di dalam paket

**`models/stc_rppg/__init__.py`** harus mengekspor kelas rakitan:

    from .model import STCrPPG            # atau .stc_rppg jika nama file dipertahankan
    from .ditac import DiTAC
    from .loss import STCHybridLoss

## Catatan

- Agar `data.preprocessing` & `data.loaders` bisa di-import sebagai paket, tambahkan `__init__.py`
  kosong di `data/`, `data/preprocessing/`, `data/loaders/` (atau jalankan dari root repo).
- Jalankan test dari root: `pytest tests/`.
- Ini rencana migrasi; saya belum mengeksekusi karena butuh repo lengkap. Bila Anda kirim repo
  utuh, saya bisa terapkan langsung + samakan seluruh import.
