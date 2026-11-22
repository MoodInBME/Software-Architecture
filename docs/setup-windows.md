# Lingkungan Windows dengan GPU (laptop)

Panduan ini membuat venv `.venv-gpu` dengan Python 3.12 di laptop Windows dengan GPU NVIDIA. Semua perintah dijalankan di PowerShell dari folder repo tanpa mengaktifkan venv: setiap perintah memanggil `.\.venv-gpu\Scripts\python.exe` langsung, sehingga `python` atau `python3` milik sistem tidak pernah terpakai.

## Prasyarat

- Python 3.12 terpasang dan terlihat di `py -0p`.
- Driver NVIDIA terbaru. **CUDA Toolkit tidak diperlukan**: wheel PyTorch `cu124` membawa runtime CUDA 12.4 dan cuDNN sendiri, dan driver yang lebih baru tetap menjalankan runtime yang lebih lama. Toolkit (`nvcc`) hanya dibutuhkan untuk mengompilasi ekstensi CUDA sendiri, yang tidak dilakukan repo ini.
- Ruang disk sekitar 6 GB untuk venv (wheel torch CUDA sekitar 2,5 GB).

## Langkah pemasangan

```powershell
Set-Location C:\Users\Jeremia\GEMASTIK\MoodIn_Software
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip

# 1. PyTorch CUDA 12.4 (satu-satunya langkah yang memakai indeks PyTorch)
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 2. Paket lain dengan versi terkunci (menurunkan numpy ke 1.26.4 bila langkah 1 menarik numpy 2)
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements\windows-gpu.txt -c requirements\constraints.txt

# 3. mediapipe tanpa dependensinya (mencegah OpenCV kedua dan jax)
.\.venv-gpu\Scripts\python.exe -m pip install --no-deps -r requirements\windows-nodeps.txt

# 4. Alat uji
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-dev.txt -c requirements\constraints.txt

# 5. Periksa
.\.venv-gpu\Scripts\python.exe scripts\cek_lingkungan.py
.\.venv-gpu\Scripts\python.exe -m pytest tests -q
```

Hasil yang diharapkan dari langkah 5:

- `cek_lingkungan.py` diakhiri `OK: tidak ada masalah yang dikenal.`;
- torch menampilkan `build CUDA: 12.4` dan nama GPU;
- numpy `1.26.4`, satu paket OpenCV (`opencv-contrib-python`), mediapipe `0.10.14 (solutions: True)`;
- backend kamera OpenCV memuat `DSHOW` dan `MSMF`.

`pip check` akan melaporkan bahwa mediapipe meminta `jax`, `jaxlib`, dan `sounddevice`. Itu disengaja: `mp.solutions.face_mesh` dan `mp.solutions.hands` tidak mengimpor ketiganya.

Simpan daftar paket setelah lingkungan sehat, di luar repo:

```powershell
New-Item -ItemType Directory -Force D:\moodin_runs\lingkungan | Out-Null
.\.venv-gpu\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 D:\moodin_runs\lingkungan\freeze_venv-gpu.txt
```

## Jebakan yang dikenal

**torch CPU menimpa torch CUDA.**
Gejala: `cek_lingkungan.py` mencetak `build CUDA: None`, `torch.cuda.is_available()` bernilai False.
Pencegahan: jangan `pip install -U`, jangan memasang paket yang meminta torch tanpa `-c requirements\constraints.txt`, dan jangan memasang `requirements.txt` ke `.venv-gpu` (berkas itu juga memuat `opencv-python`).
Pemulihan:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install --force-reinstall --no-deps torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

**numpy 2.**
Gejala: mediapipe atau OpenCV gagal diimpor dengan pesan modul dikompilasi untuk NumPy 1.x; `cek_lingkungan.py` melaporkan numpy bukan 1.26. Pemasangan torchvision dari indeks cu124 menarik numpy terbaru.
Pencegahan: selalu memasang dengan `-c requirements\constraints.txt`.
Pemulihan: `.\.venv-gpu\Scripts\python.exe -m pip install numpy==1.26.4`.

**mediapipe 1.x.**
Gejala: `AttributeError: module 'mediapipe' has no attribute 'solutions'`. Versi 1.x hanya punya API Tasks, sedangkan `inference.py` memakai `solutions.face_mesh`.
Pemulihan: `.\.venv-gpu\Scripts\python.exe -m pip install --no-deps mediapipe==0.10.14`.

**OpenCV ganda.**
Gejala: `cek_lingkungan.py` mencetak lebih dari satu paket OpenCV; `cv2` berasal dari paket yang salah sehingga fungsi GUI atau modul contrib hilang. Penyebab umum: mediapipe dipasang dengan dependensinya di samping `opencv-python`.
Pemulihan:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
.\.venv-gpu\Scripts\python.exe -m pip install opencv-contrib-python==4.10.0.84 -c requirements\constraints.txt
```

**dlib di Windows.**
Gejala: `pip install dlib` mencoba kompilasi dan gagal tanpa CMake dan Visual Studio Build Tools.
Pencegahan: jangan memasang dlib di `.venv-gpu`. Untuk uji pembanding landmark, pakai lingkungan terpisah (WSL2 atau conda-forge).

**`python` dan `python3` menunjuk versi lain.**
Di laptop ini `python` = 3.15 dan `python3` = 3.14. Selalu memakai `.\.venv-gpu\Scripts\python.exe` atau `py -3.12`.

**`where nvcc` tidak mencetak apa pun.**
Di PowerShell `where` adalah alias `Where-Object`. Pakai `where.exe nvcc` atau `Get-Command nvcc`.

**TensorFlow.**
TensorFlow di Python 3.14 global tidak terlihat dari venv. Jangan memasang TensorFlow di `.venv-gpu`; ia menarik numpy dan protobuf yang bentrok dengan kunci repo.

**DataLoader macet di Windows.**
Windows memakai _spawn_ untuk proses pekerja. Jalankan skrip lewat `python -m ...` yang punya `if __name__ == "__main__":`; bila macet, turunkan `num_workers` ke 0 untuk memeriksa.

## Memulihkan dari awal

Bila beberapa jebakan terjadi sekaligus, membuat ulang venv lebih cepat daripada memperbaikinya satu per satu. Folder `.venv-gpu` ada di `.gitignore` dan tidak pernah di-commit.

```powershell
Set-Location C:\Users\Jeremia\GEMASTIK\MoodIn_Software
Remove-Item -Recurse -Force .venv-gpu
```

Lalu ulangi bagian "Langkah pemasangan". Wheel torch biasanya masih ada di cache pip sehingga tidak diunduh ulang.
