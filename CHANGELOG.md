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
- Jendela jalur wajah menjadi 90 _frame_ (3 detik pada 30 fps) dengan geser 15 _frame_, sebelumnya 150 _frame_: `data.window_frames` di `models/st_gcn_graph_tran/config.yaml` dan nilai bawaan `train.py`, serta `clip_len_stgcn` di `config/default.yaml` dan `inference.py`. _Checkpoint_ ST-GCN lama tetap dapat dimuat karena bobot tidak bergantung pada panjang jendela, tetapi dilatih pada 150 _frame_.
- `communication/` mengikuti kontrak IPC 2.0.0 di MoodIn-master: field wajib `schema_version`, `session_id`, dan `ts` (integer milidetik) di semua payload; `decision_margin` menggantikan `confidence`; `simulated` dan `physio_used` ikut dikirim; arah `moodin/ui/command` (`command`) dari antarmuka ke mesin inferensi; opsi TLS dan port 8883 dihapus sesuai ADR-10; topik baru `moodin/gesture/event` (`GestureEvent`, `publish_gesture`). **[lintas repo]** Isi `presence` dan `signal/quality` belum selaras kontrak.

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
- Docstring `labels_tabel33.py` dan kartu dataset AFEW-VA serta DISFA diringkas ke tingkat tinggi; pemeriksaan folder keluaran di `afew_va.py` dan `disfa.py` memakai `Path.is_relative_to`.
- `models/st_gcn_graph_tran`: `data.mood_rule` di `config.yaml` (bawaan `tabel33`, pilihan `lingkaran` untuk arsip GEMASTIK). Pelatihan menghitung ulang label mood lewat `MoodRuleDataset` tanpa mengubah `expression_video_loader.py`; `evaluate_va` menerima `mood_fn` dan mengeluarkan sampel di celah aturan dari metrik mood; rugi klasifikasi mengabaikan celah itu. `labels.py` tetap ada.
- `models/fusion/late_fusion.py`: mood hanya dari jalur wajah dan dipetakan dengan aturan Tabel 3.3 (`mood_rule: tabel33`, `lingkaran` untuk arsip); HR dan HRV hanya diteruskan ke payload dan tidak lagi dicampur ke arousal; `physio_to_arousal` dihapus. Skor `confidence` diganti `decision_margin` (heuristik jarak ke batas kategori, bukan peluang), dengan histeresis khusus tampilan. Field MQTT `confidence` pada kontrak v1 diisi `decision_margin` sampai kontrak v2. Overlay `inference.py` menulis "Margin".
- `models/st_gcn_graph_tran/mood_rules.py`: mode `label` dan `prediction`, `boundary_distance`, `decision_margin`, dan `MoodDisplayHysteresis`; modul tidak lagi bergantung pada torch. `evaluate_va` menerima `pred_mood_fn`.
- `data/dataset_cards/label-harmonization.md` diringkas ke tingkat tinggi.
- Pilihan backend kamera `camera_backend: csi | uvc` di `config/default.yaml` (bawaan `csi`); `uvc` membuka kamera lewat V4L2 dengan YUYV 640 × 480 pada 30 fps (usulan Logitech C920, ADR-15 di MoodIn-master). Modul baru `data/acquisition/` (`camera.py`, `frame_stats.py`). **[lintas repo]** Dokumentasi di Hardware-Connections.
- `scripts/kunci_kontrol_uvc.sh` dan `config/uvc_kontrol.conf`: mengunci frekuensi jala-jala, eksposur, _white balance_, dan fokus pada kontrol yang tersedia saja. `scripts/uji_banding_kamera.py`: uji banding kamera yang hanya menulis CSV (selang antar-frame, frame hilang, kecerahan ROI per 10 detik), tanpa menyimpan frame.
- Lingkungan Windows dengan GPU: `requirements/windows-gpu.txt`, `requirements/windows-nodeps.txt` (mediapipe dipasang `--no-deps`), dan `requirements/constraints.txt` (Python 3.12; numpy 1.26.4, mediapipe 0.10.14, satu paket OpenCV; torch dipasang terpisah dari indeks cu124). `scripts/cek_lingkungan.py` mencetak versi dan backend kamera tanpa membuka kamera. Panduan `docs/setup-windows.md`; `.venv-gpu/` diabaikan git.
- Temuan audit 22 Nov: `data/preprocessing/landmark_map.py` menjadi satu-satunya sumber peta MediaPipe -> iBUG-68 dengan dua varian (`A` untuk `inference.py`, `B` untuk ekstraksi dataset; berbeda di titik 0, 17, 26), menunggu uji NME. `data/preprocessing/au_scale.py`: konversi skala AU 0..5 <-> 0..1 yang menolak skala salah; `au_report` kini menolak masukan 0..5. `read_openface_image_csv` dan `best_face` untuk CSV OpenFace FaceLandmarkImg. `AffectLandmarkDataset(return_frame_mask=True)` dan `STGCN_GraphTransformer.forward(x, frame_mask)`: frame padding diabaikan Transformer dan _pooling_ temporal. `scripts/validasi/laporan_padding_afew.py`.
- `models/stc_rppg/ditac.py` ditulis ulang mengikuti persamaan DiTAC di paper rujukan (keluaran x̃·Φ(x), T monoton linear sepotong-sepotong, setara GELU saat inisialisasi). Bentuk parameter `theta_raw` sama, tetapi _checkpoint_ STC-rPPG lama dilatih dengan aktivasi versi lama sehingga keluarannya berbeda. `tests/test_ditac.py` ditulis ulang sebagai uji pytest; `tests/main.py` diganti `tests/test_stcrppg_shape.py`.
- Modul gestur tangan `interaction/` (usulan, ADR-16 di MoodIn-master): fitur tangan ternormalisasi, cubit dengan histeresis dua ambang, filter One Euro, mesin keadaan diam/menunjuk/memilih, telapak ditahan 1 detik untuk kembali, nonaktif selama fase ukur, dan pesan `moodin/gesture/event`. `scripts/demo_gestur.py` untuk webcam laptop (tanpa menyimpan frame; `--mqtt` opsional). Uji `tests/test_gestur.py` memakai titik sendi sintetis.
- Validasi lokal (sebagian): `docs/jejak-diagram.md` (blok diagram arsitektur ke kode, uji, dan metrik); `scripts/validasi/` berisi V1 NME 68 landmark pada 300-W, V3 CHROM pada UBFC-Phys terhadap BVP kontak (video dibaca streaming), V5 ekspor ONNX opset 17 beserta skrip TensorRT dan kalibrasi INT8 untuk Jetson, dan laporan padding AFEW-VA. `data/preprocessing/chrom.py` sebagai pembanding rPPG klasik. Semua keluaran wajib di luar repo.
- `docs/cloud-training.md`: cloud (Kaggle, Colab lewat WSL2) hanya untuk data sintetis atau dataset yang lisensinya mengizinkan; larangan untuk Aff-Wild2, DISFA, video UBFC, dan data partisipan. `notebooks/kaggle_latih_stgcn.ipynb` ditandai DIBLOKIR sampai lisensi AFEW-VA dicek.
- Windows: filter One Euro tidak lagi berhenti dengan galat bila cap waktu tidak naik (resolusi `time.monotonic` sekitar 15,6 ms); nilai terakhir dikembalikan dan kejadiannya dihitung. Demo gestur dan uji banding kamera memakai `time.perf_counter`. Tiga uji `kunci_kontrol_uvc.sh` dilewati di Windows. `pyproject.toml` memberi `pythonpath = ["."]` untuk pytest.

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
