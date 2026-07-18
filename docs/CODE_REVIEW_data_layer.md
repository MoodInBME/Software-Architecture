# Telaah Data-Layer + Rekonsiliasi README (branch `main`)

Telaah file: `data/dataset_cards/ubfc-rppg.md`, `data/loaders/{dataset_loader.py,
dataset_ubfc_loader.py, prepare_data_ubfc.py}`, dan `README.md` root teman Anda.
Label keyakinan: **[Tinggi] / [Sedang] / [Rendah]**. Kode tak dieksekusi (torch/opencv/
mediapipe tidak terpasang di sandbox); validasi hanya `py_compile`.

---

## D0 [KRITIS - KEAMANAN] Token Kaggle bocor

`ubfc-rppg.md` dan `dataset_ubfc_loader.py` berisi **API token Kaggle** ter-hardcode
(`KGAT_fbaa...f6e7`). Jika sudah ter-commit/ter-push (repo publik), token itu **bocor permanen**
- menghapus file TIDAK cukup karena tetap ada di riwayat git.

**Tindakan wajib:**
1. **Cabut/rotasi** token di akun Kaggle (Account -> API -> Expire/Create New Token) **sekarang**.
2. Hapus token dari kode; simpan kredensial di `~/.kaggle/kaggle.json` (chmod 600) atau env
   `KAGGLE_USERNAME`/`KAGGLE_KEY`. Lihat `download_ubfc_kaggle.py` (pengganti tanpa token).
3. Tambahkan `kaggle.json`, `*.pt`, `dataset/` ke `.gitignore`.
4. (Opsional) bersihkan riwayat dgn `git filter-repo` bila token pernah masuk history.

---

## D1 [Tinggi] `ubfc-rppg.md` salah isi

Itu skrip unduh Python, bukan kartu dataset. Diganti dengan kartu dataset benar
(lisensi, struktur, sitasi, cara pakai). Skrip unduh -> `download_ubfc_kaggle.py`.

## D2 [Tinggi] `dataset_loader.py` kosong (0-2 byte)

Belum ada implementasi. Ditambahkan `TensorClipDataset` generik (memuat .pt {video,target}).

## D3 [Tinggi] `dataset_ubfc_loader.py` bukan loader

Versi lama hanya skrip unduh Kaggle (duplikat + token). Diganti `UBFCrPPGDataset` sebenarnya:
windowing jendela-geser, sinkronisasi PPG, edge-pad (bukan nol), normalisasi target untuk
Neg-Pearson, `weights_only=True`.

## D4 [Tinggi] `prepare_data_ubfc.py`: label PPG hilang + isi dummy

Masalah versi lama:
- Hanya menyimpan `torch.randn(3,128,128,128)` (dummy) - **tidak memproses wajah**.
- **Tidak menyimpan ground-truth PPG** -> STC-rPPG tak punya target latih (fatal).
- Path Colab `/content/...` di-hardcode; `base_name = split('.')[0]` rusak utk nama ber-titik.

Versi baru: crop wajah MediaPipe nyata, simpan `{video,(C,T,H,W), ppg(T,), hr_mean, fps}`,
PPG diresample ke jumlah frame, path via `argparse`, nama file aman.

## D5 [Sedang] `.split('.')[0]` & path hardcode

Diperbaiki di D4 (`os.path.normpath` + `basename`, argumen CLI).

---

## Rekonsiliasi README (2 versi)

Ada **dua README** kini: milik teman Anda (`README.md`) dan draf saya sebelumnya. Perbedaan penting:

1. **Target edge berbeda ⚠️ [Tinggi]** - README teman menyebut **Raspberry Pi 5**
   (`deployment/edge_inference_rpi5.py`), sedangkan Anda sedang membeli **Jetson Orin Nano Super**.
   Ini harus diputuskan: satu board kanonik. Jika Jetson dipilih, ubah README + rencana
   `deployment/` (tambah `edge_inference_jetson.py`, jalur ONNX->TensorRT), dan turunkan Pi 5 jadi
   opsi cadangan.
2. **Nama env** - teman: `moodin-ml`; draf saya: `moodin`. **Selaraskan ke `moodin-ml`**
   (sudah saya samakan di `environment.yml` paket ini). `environment.yml` = sumber kanonik,
   `requirements.txt` diturunkan darinya.
3. **Struktur (drift) [Tinggi]** - README teman menampilkan STC-rPPG **hanya** di `models/stc_rppg/`
   (tanpa file komponen datar `ditac.py/fsfe.py/...` di root `models/`). Ini justru sesuai
   reorganisasi awal saya (komponen di dalam paket `stc_rppg/`), dan **menghindari tabrakan nama**
   `stc_rppg.py` vs `stc_rppg/` yang saya soroti (S1). Artinya: tata letak DATAR yang Anda kirim
   sebelumnya adalah kondisi repo aktual yang menyimpang dari README. **Rekomendasi:** ikuti
   struktur README (komponen di dalam `models/stc_rppg/`), buang file datar duplikat.
4. **Konsistensi arsitektur** - README teman sendiri menandai ambiguitas: apakah dual-head
   melekat pada ST-GCN atau pada modul fusi (juga kelemahan W3). Samakan satu versi di README,
   kode fusi, dan Bab 2.5 proposal sebelum evaluasi.
5. **`label-harmonization.md`** disebut di README tetapi belum ada -> disertakan di paket ini.
6. Sitasi (Lin 2025, Zhao 2023, Antoniadis 2021) masih ditandai "verifikasi" - konfirmasikan
   sebelum masuk laporan.

---

## Prioritas tindakan

1. **D0** - cabut token Kaggle (darurat).
2. **D4/D3/D2** - pasang data-layer yang menyimpan & memuat PPG (tanpa ini STC-rPPG tak bisa dilatih).
3. **Rekonsiliasi #1** - putuskan board edge (Pi 5 vs Jetson) lalu selaraskan README + deployment.
4. **Rekonsiliasi #3** - rapikan struktur `models/` (buang duplikat datar).
5. `data/preprocessing/{clahe_roi.py, landmark_extraction_mediapipe.py}` masih 0 KB - perlu diisi
   (bisa saya buatkan; `preprocessing_mediapipe.py` dari paket sebelumnya bisa jadi basis).
