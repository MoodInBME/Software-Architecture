# Telaah File Root Software-Architecture (branch `main`)

Telaah kritis atas 6 file root yang diunggah teman Anda: `main.py`, `inference.py`,
`preprocessing.py`, `postprocessing.py`, `train.py`, `test_ditac.py`, ditambah
evaluasi struktur repo baru.

Setiap temuan diberi label keyakinan: **[Tinggi]**, **[Sedang]**, **[Rendah]**.
Catatan: kode **tidak dapat dieksekusi** di sandbox (PyTorch/OpenCV/MediaPipe tidak
terpasang + jaringan mati). Validasi hanya sebatas sintaks (`py_compile`) dan analisis statis.

---

## S1 [Tinggi] Tabrakan nama: `models/stc_rppg.py` vs folder `models/stc_rppg/`

Struktur baru mencantumkan **keduanya** di dalam `models/`:

    models/
      stc_rppg/        <- FOLDER (checkpoints/, config.yaml, model.py, train.py)
      stc_rppg.py      <- FILE  (kelas STCrPPG hasil rakitan)

Ini **tidak sah** untuk sistem import Python. Saat menulis `from models.stc_rppg import STCrPPG`
(dipakai di `inference.py`), resolver Python mengutamakan **paket** (`models/stc_rppg/`)
di atas **modul** (`stc_rppg.py`). Karena folder `stc_rppg/` tidak mendefinisikan `STCrPPG`,
import akan `ImportError` atau menunjuk objek yang salah. Ini bug diam-diam yang berbahaya.

Catatan konteks: pada percakapan sebelumnya saya memindahkan 7 file komponen ke dalam
`models/stc_rppg/`. Ternyata teman Anda memilih tata letak **datar** (komponen di root `models/`).
Saya **selaraskan ke pilihan tim** (datar), tetapi tabrakan nama tetap harus diselesaikan.

**Rekomendasi (pilih salah satu):**

- **Opsi A (churn minimal, disarankan):** ganti nama file rakitan `models/stc_rppg.py`
  menjadi `models/stc_rppg_net.py`. Lalu:
  - `models/__init__.py`: `from .stc_rppg_net import STCrPPG`
  - `inference.py`: `from models import STCrPPG` (lewat `__init__`, bukan `from models.stc_rppg import ...`)
- **Opsi B (paling rapi arsitektural):** pindahkan komponen STC-rPPG (`ditac.py`, `fsfe.py`,
  `stfe.py`, `predictor.py`, `loss.py`, rakitan `stc_rppg.py`) ke dalam paket `models/stc_rppg/`,
  karena semuanya khusus STC-rPPG. Namun ini mengubah banyak jalur import dan berbeda dari
  keputusan tim, jadi hanya jika tim setuju.

Selesaikan ini **sebelum** menambah `models/stc_rppg/model.py` agar tidak mewarisi ambiguitas.

---

## S2 [Tinggi] Inkonsistensi pipeline wajah: dlib vs MediaPipe

`preprocessing.py` memakai **dlib** (`dlib.get_frontal_face_detector()`), sedangkan struktur
baru punya `data/preprocessing/landmark_extraction_mediapipe.py` (MediaPipe). Dua jalur
deteksi wajah berbeda = sumber kebenaran ganda.

Masalah dlib untuk proyek ini:

1. **Berat dipasang** (butuh CMake + toolchain C++), khususnya di ARM (Jetson/Pi) - rawan gagal build.
2. **Hanya bounding box**, tanpa landmark. Padahal cabang **ST-GCN butuh koordinat landmark**.
   Berarti Anda tetap perlu detektor kedua -> pemborosan komputasi realtime.
3. Detektor HOG dlib lemah pada **cahaya rendah + two-way mirror + (calon) NoIR** (kelemahan W1).

**Rekomendasi:** standarkan ke **MediaPipe FaceMesh** sebagai satu-satunya detektor. Satu proses
menghasilkan **ROI wajah (untuk STC-rPPG)** sekaligus **landmark (untuk ST-GCN)**. Saya sertakan
drop-in `suggested/preprocessing_mediapipe.py` (lihat di bawah). Hapus dependensi `dlib`.

---

## S3 [Sedang-Tinggi] Resolusi BPM `postprocessing.py` terlalu kasar

Dengan `N=128` frame @ `fps=30`, resolusi frekuensi FFT:

    df = fps / N = 30 / 128 = 0.234 Hz  ->  ~14 BPM per bin

Artinya estimasi detak jantung hanya bisa "melompat" ~14 BPM. Untuk klinis/wellness ini kasar.
Ditambah `np.argmax(fft_mag)` mencari di **seluruh** spektrum (termasuk sisa DC/tren rendah),
bukan dibatasi ke pita HR.

**Perbaikan (ada di `suggested/postprocessing.py`):**
- Batasi pencarian puncak ke pita HR `[0.7, 3.5] Hz`.
- **Zero-pad** FFT (`n_fft=4096`) + jendela Hanning -> resolusi jauh lebih halus & kebocoran spektral turun.
- **Interpolasi parabola** di sekitar puncak untuk presisi sub-bin.
- Ganti `filtfilt` order-5 -> **`sosfiltfilt`** (second-order sections) agar stabil secara numerik.

Kaitan: ini juga menyentuh kelemahan W5/W6 (jendela pendek buruk untuk HR/HRV). Untuk HRV,
naikkan panjang jendela (mis. >=300 frame / >=10 s).

---

## S4 [Sedang] `inference.py`: `torch.load` tanpa `weights_only`

`torch.load(weight_path, map_location=device)` memuat pickle penuh. Sejak PyTorch 2.6
default berubah ke `weights_only=True`, dan memuat pickle sembarang adalah risiko keamanan.

**Perbaikan:** `torch.load(weight_path, map_location=device, weights_only=True)` dan simpan
hanya `state_dict` saat training (`torch.save(model.state_dict(), ...)`). Sudah diterapkan di
`suggested/inference.py`.

---

## S5 [Sedang] `train.py` masih kerangka + bug bentuk tensor + entrypoint ganda

1. Banyak placeholder (`self.video_paths=[...]`, `target_tensor = ...`), variabel tak terdefinisi
   (`num_epochs`, `device`), tidak ada model/optimizer/loss -> **tidak bisa dijalankan**. Wajar
   untuk stub, tapi harus dilengkapi.
2. **Bug bentuk:** `FaceVideoProcessor.process_video()` mengembalikan `(1, C, T, H, W)` (sudah
   ada dimensi batch). Di `Dataset.__getitem__`, DataLoader akan menambah batch lagi -> jadi
   `(B, 1, C, T, H, W)` (6D, salah). `__getitem__` harus mengembalikan `(C, T, H, W)` tanpa batch.
   Versi `suggested/preprocessing_mediapipe.py` mengembalikan `(C, T, H, W)` sehingga memperbaiki ini.
3. **Entrypoint training ganda:** ada `train.py` root **dan** `models/stc_rppg/train.py` **dan**
   `models/st_gcn_graph_tran/train.py`. Tetapkan hierarki: `train.py` root = **orkestrator**
   (parsing arg `--model stc_rppg|st_gcn|fusion`, baca config) yang **memanggil** trainer per-model.
   Jangan duplikasi loop training.
4. Path `/content/dataset_aktif/` (Colab) di-hardcode -> pindahkan ke `config/default.yaml`.

---

## S6 [Sedang] Zero-padding frame menyuntik artefak spektral

`preprocessing.py` mengisi kekurangan frame dengan **nol** (`np.zeros`). Untuk sinyal rPPG,
blok nol = diskontinuitas tajam -> kebocoran spektral -> estimasi BPM bias. Lebih baik
**ulangi/refleksikan** frame terakhir, atau **tolak** klip yang terlalu pendek. Diterapkan
(`pad_mode="loop"`/`"edge"`) di versi MediaPipe.

---

## S7 [Sedang] `test_ditac.py` bagus, tapi bergantung asumsi arsitektur

Tes ini benar secara metodologis (cek: forward, identitas di luar `[-3,3]`, aliran gradien ke
`ditac.theta` dan ke input). Namun ia hanya lolos jika `theta` adalah `nn.Parameter` leaf.
Ini **tidak** memvalidasi dua isu yang saya angkat di telaah STC-rPPG sebelumnya:
- `theta` belum dibatasi **monoton** -> DiTAC belum benar-benar diffeomorphic.
- Blok `cdc_st_mock` / `ditac_mock` masih `Conv3d` biasa (klaim CDC belum terpenuhi).

Tambahkan tes monotonisitas `theta` bila Opsi patch DiTAC (cumsum-softmax) diadopsi.

---

## S8 [Rendah] Penempatan file & penamaan

- `test_ditac.py` ada di root, tapi `tests/` sudah punya `test_models.py`, `test_preprocessing.py`.
  Konsolidasikan semua tes di `tests/` (mis. `tests/test_ditac.py`) agar `pytest` konsisten.
- `main.py` sebenarnya adalah **smoke test** `STCrPPG`, bukan entrypoint aplikasi. Pertimbangkan
  memindah ke `tests/test_models.py` atau menamainya `smoke_test.py`. Klaim "~0.82M param"
  belum terverifikasi (tidak bisa dijalankan di sini) - verifikasi di lingkungan tim.
- `models/stc_rppg/checkpoints/` dan `st_gcn_graph_tran/checkpoints/` kosong: tambahkan
  `.gitkeep` + pastikan `.gitignore` mengabaikan `*.pth`/`*.onnx` besar (gunakan Git LFS/DVC).

---

## S-ENV [Tinggi] Versi Python & Torch

- **Python 3.14 dihindari:** wheel PyTorch & MediaPipe belum tersedia; MediaPipe 0.10.x resmi
  mendukung 3.9-3.12. Ini menutup opsi 3.14.
- **Disarankan Python 3.10** sebagai versi utama karena **selaras JetPack 6.2** (Jetson =
  Ubuntu 22.04 = Python 3.10) -> kode yang sama jalan di laptop & Jetson. 3.11/3.12 boleh di laptop.
- **Torch di Jetson berbeda:** wajib wheel resmi NVIDIA (jp6), bukan PyPI. Lihat `requirements-jetson.txt`.

---

## Ringkasan prioritas tindakan

1. **S1** - selesaikan tabrakan `stc_rppg.py` vs `stc_rppg/` (Opsi A).
2. **S2** - satukan ke MediaPipe, buang dlib.
3. **S3** - pasang postprocessing BPM yang lebih presisi.
4. **S5** - perbaiki bentuk tensor Dataset + tetapkan orkestrator training.
5. **S4/S6/S7/S8** - keamanan `torch.load`, padding, konsolidasi tes.

File perbaikan siap-pakai ada di folder `suggested/`. Ini **usulan**, bukan penimpaan otomatis;
setelah Anda kirim repo lengkap di percakapan berikutnya, saya bisa terapkan langsung + samakan
dengan `data/loaders/*` dan `data/preprocessing/*`.
