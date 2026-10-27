# Changelog — Software-Architecture

Perubahan penting pada repo ini dicatat di sini. Perubahan yang berdampak ke repo lain ditandai **[lintas repo]**.

Format mengikuti [Keep a Changelog](https://keepachangelog.com/id/1.1.0/).

## [Unreleased] — fase Tugas Akhir

### Diubah

- `README.md` ditulis ulang untuk fase TA: ruang lingkup per jalur, cara menjalankan, tabel dataset, "Asal kode", dan rujukan.
- `data/dataset_cards/ubfc-rppg.md` diberi status pada fase TA.
- `data/dataset_cards/label-harmonization.md` dan `models/st_gcn_graph_tran/README.md` diberi catatan bahwa isinya versi era GEMASTIK.
- Rujukan path di komentar `data/preprocessing/landmark_extraction_mediapipe.py` dan `models/st_gcn_graph_tran/labels.py`, serta pesan petunjuk di `train.py`, disesuaikan dengan lokasi arsip yang baru.

### Ditambahkan

- Kartu dataset `data/dataset_cards/ubfc-phys.md`. Kartu `pure.md` yang sebelumnya kosong kini berisi.
- `CHANGELOG.md` dan `docs/arsip-gemastik/README.md`.

### Dipindah

- Dokumen kerja era GEMASTIK di `docs/` ke `docs/arsip-gemastik/`.
- Notebook CK+ ke `notebooks/arsip-gemastik/`. Notebook hasil uji di Kaggle diberi nama tanpa spasi (`ck_plus_stgcn_kaggle_hasil_uji_kaggle.ipynb`).
- Kedua notebook kini meng-_clone_ repo pada tag `gemastik-xix-2026`, bukan branch default, agar tetap cocok dengan kode era itu.

### Dihapus

- 12 berkas _placeholder_ kosong (0 byte): tiga notebook yang tidak valid, `evaluation/metrics.py`, `data/preprocessing/clahe_roi.py`, kartu dataset AffectNet dan CK+, tiga berkas `evaluation/results/*.json`, `tests/test_models.py`, dan `tests/test_preprocessing.py`.
- Bobot dummy `weights/dummy_stgcn.pth` dan `weights/dummy_stcrppg.pth` dikeluarkan dari pelacakan git; berkasnya tetap ada di disk laptop. Buat ulang dengan `python generate_dummy.py`.
  - **[lintas repo]** Klon di Jetson akan kehilangan kedua berkas itu saat `git pull`. Jalankan `python3 generate_dummy.py` sesudahnya.

## [gemastik-xix-2026] — 2026-10-27

Arsip akhir era GEMASTIK XIX 2026 (tim VisionaryX). Tag menunjuk commit `43b4756` (8 Agustus 2026).

Isi utama:

- STC-rPPG beserta pemuat dan skrip persiapan UBFC-rPPG;
- cabang ekspresi ST-GCN + Graph Transformer pada 68 _landmark_ yang dipetakan dari MediaPipe 468;
- fusi _late fusion_ valensi-_arousal_;
- klien MQTT dengan kontrak v1.0.0;
- `inference.py` untuk Jetson dengan mode demo;
- notebook pelatihan CK+ di Kaggle;
- skrip diagnosis kamera dan daya untuk Jetson.

[Unreleased]: https://github.com/MoodInBME/Software-Architecture/compare/gemastik-xix-2026...HEAD
[gemastik-xix-2026]: https://github.com/MoodInBME/Software-Architecture/tree/gemastik-xix-2026
