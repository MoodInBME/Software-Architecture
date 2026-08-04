# Panduan Setup Jetson Orin Nano Super (MoodIn Edge)

Target: **NVIDIA Jetson Orin Nano Super (8 GB)**, JetPack 7.2 (R39.2.0),
Ubuntu 24.04, CUDA 13.2.1, Python 3.12.

---

## 1. Prasyarat OS

Pastikan Jetson sudah di-flash dengan JetPack 7.2 via NVIDIA SDK Manager.
Verifikasi:

```bash
cat /etc/nv_tegra_release    # Harus: R39 (release), REVISION: 2.0
python3 --version            # Harus: 3.12.x
nvcc --version               # Harus: CUDA 13.2 atau 13.3
```

Jika versi tidak sesuai (terlalu tertinggal), **jangan lanjutkan** dan flash ulang dulu. 
*(Catatan: Jika `nvcc` Anda menunjukkan **CUDA 13.3**, ini sangat normal untuk update JetPack 7.2 terbaru. Wheel PyTorch `cu132` sepenuhnya kompatibel dan akan berjalan tanpa masalah di CUDA 13.3 berkat kompatibilitas mundur minor version dari driver CUDA).*

---

## 2. Clone Repo dan Virtual Environment

Pertama, pastikan Anda men-clone repo terlebih dahulu sebelum membuat *virtual environment*:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libopenblas-dev \
    libhdf5-dev libjpeg-dev zlib1g-dev libavcodec-dev libavformat-dev \
    libswscale-dev cmake mosquitto mosquitto-clients

cd ~
git clone https://github.com/MoodInBME/Software-Architecture.git MoodIn_Software
cd MoodIn_Software
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## 3. Instalasi PyTorch (GPU, CUDA 13.2)

**KRITIS: Jangan gunakan `pip install torch` tanpa `--index-url`.**
Versi PyPI default adalah CPU-only x86_64 dan akan gagal di ARM64.

```bash
pip install torch==2.12.0+cu132 --index-url https://download.pytorch.org/whl/cu132
```

Verifikasi:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Output yang diharapkan: True  Orin (atau nama GPU Jetson)
```

Jika `torch.cuda.is_available()` mengembalikan `False`, kemungkinan Anda
menginstal wheel yang salah. Coba cari wheel alternatif di:
- https://pypi.jetson-ai-lab.io/
- https://developer.nvidia.com/embedded/jetpack

---

## 4. Instalasi torchvision (Kompilasi dari Source)

Wheel `torchvision` untuk `cu132` + `aarch64` belum tentu ada di PyPI.
Cara paling aman adalah kompilasi dari source:

```bash
git clone --branch v0.27.0 --depth 1 https://github.com/pytorch/vision.git
cd vision
export BUILD_VERSION=0.27.0
python3 setup.py install
cd ..
rm -rf vision
```

> Catatan: proses ini memakan waktu 10-20 menit di Jetson.

---

## 5. MediaPipe (MUNGKIN GAGAL di ARM64)

MediaPipe **tidak secara konsisten** menyediakan wheel untuk `aarch64` Linux.
Coba dulu:

```bash
pip install mediapipe==0.10.14
```

### Jika gagal (kemungkinan besar):

**Opsi A: Build dari source (memerlukan Docker + Bazel)**

Ini kompleks dan memakan waktu. Lihat:
https://developers.google.com/mediapipe/framework/getting_started/install#building_on_arm

**Opsi B: Gunakan community wheel dari PINTO0309**

```bash
# Cek apakah tersedia untuk cp312 di:
# https://github.com/PINTO0309/mediapipe-bin
# Jika tidak ada cp312, coba cp311 dengan pyenv
```

**Opsi C (Recommended): Fallback ke dlib**

`inference.py` sudah dirancang untuk fallback otomatis ke dlib jika MediaPipe
tidak tersedia. Dlib mengekstrak 68 landmark iBUG secara langsung (tanpa
pemetaan 468 &rarr; 68), sehingga hasilnya justru lebih presisi.

```bash
sudo apt-get install -y cmake libboost-python-dev
pip install dlib
```

Lalu unduh model prediktor dlib:

```bash
mkdir -p weights
wget -O weights/shape_predictor_68_face_landmarks.dat.bz2 \
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 weights/shape_predictor_68_face_landmarks.dat.bz2
```

**Opsi D: Tanpa landmark sama sekali**

Jika MediaPipe dan dlib keduanya gagal, `inference.py` akan menggunakan
OpenCV Haar Cascade untuk deteksi wajah saja (tanpa landmark). STC-rPPG
masih bisa jalan (hanya butuh crop wajah), tapi ST-GCN tidak akan
menerima data dan akan menampilkan "Menunggu...".

---

## 6. Dependensi Lainnya

```bash
pip install opencv-python-headless numpy paho-mqtt pyyaml scikit-learn
```

> Gunakan `opencv-python-headless` jika Jetson Anda headless (tanpa monitor).
> Jika Anda mau menampilkan jendela OpenCV (yang diperlukan untuk demo),
> gunakan `opencv-python` atau pastikan Jetson sudah punya display.

Atau jika OpenCV sudah terinstal via JetPack (`cv2` system-level):

```bash
python3 -c "import cv2; print(cv2.__version__)"
# Jika berhasil, tidak perlu pip install opencv-python
```

---

## 7. Verifikasi dan Jalankan

*(Pastikan Anda berada di dalam folder `~/MoodIn_Software` dan `.venv` sudah aktif)*

```bash

# Verifikasi semua file penting ada
python3 -c "
from models.st_gcn_graph_tran.model import STGCN_GraphTransformer
from models.stc_rppg.model import STCrPPG
from models.fusion.late_fusion import LateFusionVABridge
from data.preprocessing.landmark_map import MEDIAPIPE_FACEMESH_TO_IBUG68
print('Semua modul berhasil diimpor.')
"

# Jalankan demo
python3 inference.py
```

---

## 8. Troubleshooting Umum Jetson

| Gejala | Penyebab | Solusi |
|---|---|---|
| `torch.cuda.is_available() == False` | Wheel PyTorch salah (CPU-only) | Instal ulang dengan `--index-url` |
| `nvarguscamerasrc` error | Driver Arducam belum terinstal | Instal Arducam Jetson driver dari GitHub Arducam |
| `cannot open shared object file: libcudnn` | cuDNN tidak terhubung ke venv | `export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH` |
| `Segmentation fault` saat import mediapipe | Wheel tidak kompatibel ARM64 | Gunakan dlib (Opsi C di atas) |
| MQTT connection refused | Broker belum aktif | `sudo systemctl start mosquitto` |
| Frame hitam dari kamera CSI | Pipeline GStreamer salah | Tes dulu: `gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! videoconvert ! autovideosink` |
| `CUDA out of memory` | Model terlalu besar untuk 8 GB | Kurangi resolusi rPPG (`RPPG_H=64, RPPG_W=64`) |

---

## 9. Catatan Keamanan

- **Jangan pernah** meng-commit token, API key, atau kredensial ke dalam repo.
- File `.env` yang berisi variabel sensitif harus ada di `.gitignore`.
- Raw gambar wajah **tidak boleh keluar dari Jetson** (UU PDP No. 27/2022).
  Hanya data numerik (mood, BPM, HRV) yang boleh dipublikasikan via MQTT.
