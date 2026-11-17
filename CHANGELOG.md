# Changelog — Software-Architecture

Perubahan penting pada repo ini dicatat di sini. Perubahan yang berdampak ke repo lain ditandai **[lintas repo]**.

Format mengikuti [Keep a Changelog](https://keepachangelog.com/id/1.1.0/).

## [Unreleased] — tahap Pra-TA

### Diubah

- `README.md` ditulis ulang untuk fase TA: ruang lingkup per jalur, cara menjalankan, tabel dataset, "Asal kode", dan rujukan.
- `data/dataset_cards/ubfc-rppg.md` diberi status pada fase TA.
- `data/dataset_cards/label-harmonization.md` dan `models/st_gcn_graph_tran/README.md` diberi catatan bahwa isinya versi era GEMASTIK.
- `README.md` ditata ulang (Fase B2): banner terang/gelap, kartu dua jalur pemrosesan, dan narasi yang lebih deskriptif. Istilah diselaraskan dengan tahap Proposal Tugas Akhir (Pra-TA); "kontrak versi TA" menjadi "kontrak pesan v2".
- `data/dataset_cards/ubfc-phys.md`, `ubfc-rppg.md`, `label-harmonization.md`, `models/st_gcn_graph_tran/README.md`, dan `docs/arsip-gemastik/README.md`: istilah "fase TA" diganti "tahap Pra-TA".
- Rujukan path di komentar `data/preprocessing/landmark_extraction_mediapipe.py` dan `models/st_gcn_graph_tran/labels.py`, serta pesan petunjuk di `train.py`, disesuaikan dengan lokasi arsip yang baru.
- `README.md` (Fase C1): pelatihan di Kaggle atau Colab dibatasi pada dataset publik; data primer diambil setelah persetujuan etik KEP ITS dan tidak disimpan di repo. Bagian Lisensi menegaskan bahwa lisensi tidak mencakup dataset dan data partisipan.
- `.gitignore` (Fase C1): bagian 9 untuk folder data partisipan dan rekaman uji. **[lintas repo]** Pola yang sama ditambahkan di semua repo aktif (ADR-14 di MoodIn-master).
- Pita denyut jantung jalur fisiologis menjadi 0,75–4,0 Hz (sebelumnya 0,7–3,5 Hz) agar sama dengan rancangan Pra-TA dan paper rujukan STC-rPPG. Filter pascaproses, rugi domain frekuensi, estimasi BPM di `inference.py`, dan `models/stc_rppg/train.py` kini membaca satu konstanta yang sama.
- `models/stc_rppg/config.yaml` tidak lagi memuat `loss.freq_low`, `loss.freq_high`, `postprocessing.lowcut`, dan `postprocessing.highcut`. Config yang masih memuat kunci itu ditolak `train.py` dengan pesan galat. Contoh perintah di docstring `train.py` disesuaikan dengan lokasi folder `models/`.
- Estimasi BPM disatukan di `rPPGProcessor.estimate_bpm()` (filter lalu puncak spektrum ber-_zero-padding_). `bpm_from_signal()` di `inference.py` dan evaluasi di `models/stc_rppg/train.py` memanggilnya, sehingga BPM di cermin tidak lagi terkuantisasi sekitar 14 bpm pada klip 128 frame.
- Rugi domain frekuensi STC-rPPG kini bawaan `mae_hr`: selisih mutlak HR prediksi dan HR acuan (bpm), dengan HR prediksi dari _soft-argmax_ spektrum agar dapat diturunkan. Rugi KL-PSD versi lama tetap tersedia lewat `loss.freq_loss: kl_psd` di `models/stc_rppg/config.yaml`. Konfigurasi rugi dibangun lewat `build_criterion()` di `train.py`.
- `models/stc_rppg/config.yaml`: `training.lr` AdaDelta menjadi 1,0 sesuai paper rujukan STC-rPPG (sebelumnya 0,0001); `training.epochs` tetap 30. `train.py` menolak melanjutkan _checkpoint_ yang dilatih dengan lr berbeda dari config, karena memuat state optimizer ikut memulihkan lr lama.
- `models/stc_rppg/train.py`: rugi domain waktu dan domain frekuensi dicatat terpisah per epoch, di layar dan di `riwayat_pelatihan.csv` dalam folder _checkpoint_ (beserta lr dan jenis rugi frekuensi).

### Ditambahkan

- Kartu dataset `data/dataset_cards/ubfc-phys.md`. Kartu `pure.md` yang sebelumnya kosong kini berisi.
- `CHANGELOG.md` dan `docs/arsip-gemastik/README.md`.
- `.github/readme/`: gambar README (banner, ikon, wajah suasana hati) beserta `KREDIT.md`. **[lintas repo]** Bagian dari Fase B2 di semua repo.
- `models/stc_rppg/constants.py` (`HR_BAND_HZ`) sebagai satu-satunya tempat pita denyut jantung.
- `tests/test_hr_band.py`: uji respons filter, estimasi BPM, bin pita pada rugi, dan penolakan kunci config lama. Jalankan dengan `python -m unittest tests.test_hr_band -v` dari root repo.
- `tests/test_loss_freq.py`: uji estimasi HR terdiferensialkan, rugi MAE HR, arah gradien, pilihan KL-PSD, dan pembacaan config.
- Modul dataset afek untuk jalur wajah: `data/loaders/afew_va.py`, `data/loaders/disfa.py`, `data/loaders/openface_csv.py` (pembanding luring), `data/loaders/affect_landmark_dataset.py`, `data/loaders/subject_split.py`, `data/preprocessing/affect_windows.py`, `evaluation/affect_metrics.py`, `models/st_gcn_graph_tran/labels_tabel33.py`, `models/st_gcn_graph_tran/au_va_heads.py`, serta kartu dataset `afew-va.md` dan `disfa.md`. Data dataset tidak disimpan di repo; skrip menolak folder keluaran di dalam repo.
- `models/st_gcn_graph_tran`: `head_type: au_va` (kepala AU dan VA sejajar, rugi bermask) di samping `va`, `class`, dan `dual`; `model.au_list` di `config.yaml`. Bawaan tetap `va`.
- `requirements-dev.txt` (pytest) dan uji `tests/test_affect_datasets.py`, `tests/test_stgcn_au_va.py`. Jalankan `python -m pytest tests -q`.

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
