# Panduan Setup Jetson Orin Nano Super (MoodIn Edge)

Panduan ini dikhususkan untuk **Jetson Orin Nano Super (8GB)** milik tim VisionaryX untuk keperluan progres 50% (Demo Real-time Landmark & UI).

## 1. Lingkungan OS & JetPack
Sesuai aturan `AGENTS.md`, Jetson Anda harus menjalankan:
- **OS:** Ubuntu 24.04
- **JetPack:** 7.2 (R39.2.0)
- **Kernel:** 6.8
- **CUDA:** 13.2.1

*(Jangan gunakan OS Debian Trixie atau JetPack 6.x. Jika OS di SD Card/NVMe Anda salah, lakukan flash ulang menggunakan NVIDIA SDK Manager).*

---

## 2. Instalasi Dependensi Python (Sangat Kritis)

Di ekosistem Jetson (ARM64), **Anda tidak boleh menggunakan `pip install torch` biasa**. Menginstal versi PyPI standar akan mengunduh versi CPU-only atau versi yang tidak terhubung dengan CUDA Jetson (TensorRT), yang akan membuat model Deep Learning berjalan dengan sangat lambat.

Buka terminal di Jetson Anda dan jalankan perintah berurutan berikut:

### Langkah A: Update Sistem
```bash
sudo apt-get update
sudo apt-get install -y libopenblas-dev python3-pip python3-dev
```

### Langkah B: Instal PyTorch Khusus Jetson (jp72)
NVIDIA menyediakan *wheel* (file `.whl`) PyTorch khusus untuk arsitektur Jetson.
```bash
# Unduh wheel PyTorch 2.3.1 untuk JetPack 7.2
wget https://developer.download.nvidia.com/compute/redist/jp/v72/pytorch/torch-2.3.1a0+410ce96a.nv24.11-cp312-cp312-linux_aarch64.whl

# Instal wheel tersebut
pip3 install torch-2.3.1a0+410ce96a.nv24.11-cp312-cp312-linux_aarch64.whl
```

### Langkah C: Instal torchvision (Kompilasi dari Source)
Karena *torchvision* sangat bergantung erat pada PyTorch, ia harus dikompilasi manual:
```bash
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.18.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.18.1
python3 setup.py install --user
cd ..
```

### Langkah D: Pustaka Pendukung & MediaPipe
Kini Anda bisa menginstal dependensi sisanya via `pip` biasa. MediaPipe sudah menyediakan wheel untuk `aarch64`.
```bash
pip3 install mediapipe==0.10.14 opencv-python paho-mqtt numpy scikit-learn
```

---

## 3. Menjalankan Demo `inference.py` (Progres 50%)

File `inference.py` telah saya desain sedemikian rupa agar dapat langsung membaca wajah Anda dan menggambar jaring futuristik (salvion.ai) **walaupun** model aslinya belum Anda latih.

### Cara Menjalankan:
1. Hubungkan kamera (IMX477 CSI atau sekadar USB Webcam) ke Jetson.
2. Buka terminal di dalam folder repo ini (`MoodIn_Software`).
3. Jalankan:
   ```bash
   python3 inference.py
   ```

### Bagaimana Script Ini Bekerja Saat Demo?
- **Kamera Otomatis:** Script akan mencoba menggunakan kamera CSI (via pipeline GStreamer `nvarguscamerasrc`). Jika gagal (misal belum dicolok), ia akan otomatis berpindah ke kamera USB (`cv2.VideoCapture(0)`).
- **Visualisasi MediaPipe:** Skrip ini menarik fungsi dari `mp_face_mesh` untuk mengekstrak titik-titik dan langsung merendernya (menggambarnya) ke frame video.
- **Koneksi UI (MQTT):** Script juga mempublikasikan prediksi *mood dummy* ke broker MQTT lokal di topik `moodin/inference/result`. Jika teman Anda sudah menyiapkan UI berbasis web lokal, UI tersebut akan bereaksi secara *real-time* saat wajah terdeteksi.

*Selamat mencoba di lab! Jangan ragu untuk mencatat error yang muncul di Jetson besok, dan berikan pesan error tersebut agar kita bisa selesaikan bersama.*
