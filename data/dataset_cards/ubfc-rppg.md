# Dataset Card - UBFC-rPPG

> **Status tahap Pra-TA:** dataset ini dipakai untuk STC-rPPG pada era GEMASTIK. Dataset utama jalur fisiologis pada tahap Pra-TA adalah UBFC-Phys ([ubfc-phys.md](ubfc-phys.md)); pemakaian UBFC-rPPG selanjutnya mengikuti keputusan berikutnya.

> **Catatan:** file lama `ubfc-rppg.md` keliru berisi **skrip Python + token Kaggle** (bukan
> dokumentasi) dan **membocorkan API token**. File ini menggantikannya dengan kartu dataset yang
> benar. Skrip unduh dipindah ke `data/loaders/download_ubfc_kaggle.py` (tanpa token).

## Ringkasan

- **Nama:** UBFC-rPPG (University of Bourgogne Franche-Comte rPPG).
- **Modality:** video wajah RGB + gelombang PPG ground-truth (kontak, oksimeter jari).
- **Tujuan di MoodIn:** melatih & memvalidasi **STC-rPPG** (estimasi HR/HRV dari video). 🔴 **Tidak** memuat label emosi/mood - jangan pakai untuk melatih klasifikasi mood. `[Tinggi]`

## Struktur (DATASET_2 - VERIFIKASI dgn unduhan aktual) `[Sedang]`

    DATASET_2/
      subject1/
        vid.avi
        ground_truth.txt
      subject3/
        ...

`ground_truth.txt` umumnya 3 baris: (0) gelombang PPG, (1) HR/BPM, (2) timestamp detik.
Parser: `data/loaders/dataset_ubfc_loader.py::load_ubfc_ground_truth`.

- ~42 subjek (DATASET_2), video ~1 menit, ~30 fps, RGB. Angka pastikan ulang setelah unduh. `[Sedang]`

## Lisensi & Akses ⚠️

- UBFC-rPPG ditujukan **khusus riset akademik**; wajib mengikuti syarat penulis asli. `[Sedang]`
- Sumber resmi: halaman riset Universitas Bourgogne Franche-Comte. Mirror Kaggle
  (`malekdinarito/ubfc-rppg-dataset`) belum tentu membawa berkas lisensi resmi -
  **verifikasi hak pakai & sitasi sebelum dipublikasikan** dalam proposal atau laporan TA. `[Tinggi]`

## Sitasi (verifikasi format akhir) `[Sedang]`

- Bobbia, Benezeth, Macwan, Dubois. "Unsupervised skin tissue segmentation for remote
  photoplethysmography", Pattern Recognition Letters, 2019.

## Cara Pakai di Repo

    # 1. Kredensial Kaggle: ~/.kaggle/kaggle.json ATAU env KAGGLE_USERNAME/KAGGLE_KEY
    python data/loaders/download_ubfc_kaggle.py --path dataset

    # 2. Prapemrosesan -> .pt per subjek (video + PPG tersinkron)
    python data/loaders/prepare_data_ubfc.py \
        --dataset_dir dataset/DATASET_2 --output_dir data/prepared/ubfc

    # 3. Latih (via Dataset)
    #   from data.loaders.dataset_ubfc_loader import UBFCrPPGDataset
    #   ds = UBFCrPPGDataset('data/prepared/ubfc', clip_len=128, stride=64)
