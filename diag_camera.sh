#!/usr/bin/env bash
# MoodIn - Diagnostik kamera CSI IMX477 di Jetson Orin Nano Super
# Menjawab: kenapa NvBufSurfaceFromFd Failed dan inference.py keluar sendiri.
#
# Pakai:  bash diag_camera.sh 2>&1 | tee /tmp/diag_camera.log
# Skrip ini HANYA membaca dan menguji. Tidak ada yang diubah permanen,
# kecuali restart layanan nvargus-daemon pada LANGKAH 3 yang memang aman.

set -u
SEP() { echo; echo "==================== $* ===================="; }
OK()  { echo "[OK]   $*"; }
BAD() { echo "[GAGAL] $*"; }

SEP "0. Identitas sistem"
cat /etc/nv_tegra_release 2>/dev/null || echo "(berkas nv_tegra_release tidak ada)"
echo "kernel : $(uname -r)"
echo "distro : $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY | cut -d= -f2)"
echo "python : $(python3 -V)"

SEP "1. Apakah nvbuf_utils lama masih terpasang"
# Pesan 'nvbuf_utils: dmabuf_fd ... mapped entry NOT found' berasal dari API
# LAMA. JetPack 6 ke atas seharusnya memakai NvBufSurface. Kalau dua-duanya
# ada, kemungkinan besar ada sisa paket JetPack lama yang belum terhapus.
for f in libnvbuf_utils.so libnvbufsurface.so libnvbufsurftransform.so; do
  hits=$(find /usr/lib -name "$f*" 2>/dev/null)
  if [ -n "$hits" ]; then echo "$f:"; echo "$hits" | sed 's/^/    /'; else echo "$f: TIDAK ADA"; fi
done

SEP "2. Siapa yang sedang memegang kamera"
# Kalau ada proses lain yang belum melepas /dev/video0, Argus akan gagal
# memetakan buffer dan memunculkan persis pesan yang Anda lihat.
ls -l /dev/video* 2>/dev/null || echo "(tidak ada /dev/video*)"
if command -v fuser >/dev/null 2>&1; then
  echo "--- fuser /dev/video0 ---"
  fuser -v /dev/video0 2>&1 || echo "(tidak ada pemegang)"
fi
echo "--- proses python yang masih hidup ---"
ps -eo pid,etime,pcpu,rss,cmd | grep -E "[i]nference.py|[p]ython3" || echo "(tidak ada)"

SEP "3. Status dan restart nvargus-daemon"
systemctl is-active nvargus-daemon 2>/dev/null || true
echo "--- 20 baris log terakhir nvargus-daemon ---"
journalctl -u nvargus-daemon -n 20 --no-pager 2>/dev/null || echo "(journal tidak tersedia)"
echo "--- merestart nvargus-daemon ---"
sudo systemctl restart nvargus-daemon && sleep 2 && OK "nvargus-daemon direstart" || BAD "restart nvargus-daemon gagal"

SEP "4. Mode sensor menurut driver V4L2"
if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl -d /dev/video0 --list-formats-ext 2>&1 | head -40
else
  echo "(v4l2-ctl belum terpasang: sudo apt install v4l-utils)"
fi

SEP "5. Uji pipeline GStreamer MURNI, mode 1 (1920x1080 @ 60)"
# Ini menguji Argus tanpa melibatkan Python maupun OpenCV sama sekali.
# Kalau LANGKAH 5 gagal, masalahnya di driver/daemon, bukan di inference.py.
timeout 25 gst-launch-1.0 -v \
  nvarguscamerasrc num-buffers=60 sensor-id=0 sensor-mode=1 \
  ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1,format=NV12' \
  ! nvvidconv \
  ! 'video/x-raw,format=BGRx' \
  ! videoconvert \
  ! 'video/x-raw,format=BGR' \
  ! fakesink sync=false 2>&1 | tail -25
echo "kode keluar mode 1: $?"

SEP "6. Uji pipeline GStreamer MURNI, mode 0 (3840x2160 @ 30)"
timeout 25 gst-launch-1.0 \
  nvarguscamerasrc num-buffers=30 sensor-id=0 sensor-mode=0 \
  ! 'video/x-raw(memory:NVMM),width=3840,height=2160,framerate=30/1,format=NV12' \
  ! nvvidconv \
  ! 'video/x-raw,format=BGRx' \
  ! videoconvert \
  ! 'video/x-raw,format=BGR' \
  ! fakesink sync=false 2>&1 | tail -15
echo "kode keluar mode 0: $?"

SEP "7. Uji pipeline dengan penskalaan ke 1280x720 seperti di config"
# config default.yaml memakai proc_width 1280 dan proc_height 720.
# Penskalaan di dalam nvvidconv adalah titik yang sering memicu
# kegagalan pemetaan buffer.
timeout 25 gst-launch-1.0 \
  nvarguscamerasrc num-buffers=60 sensor-id=0 sensor-mode=1 \
  ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=60/1,format=NV12' \
  ! nvvidconv flip-method=2 \
  ! 'video/x-raw,width=1280,height=720,format=BGRx' \
  ! videoconvert \
  ! 'video/x-raw,format=BGR' \
  ! fakesink sync=false 2>&1 | tail -15
echo "kode keluar mode 1 + skala: $?"

SEP "8. Dukungan GStreamer di dalam OpenCV yang dipakai venv"
python3 - <<'PY'
import cv2, sys
print("opencv :", cv2.__version__)
print("berkas :", cv2.__file__)
info = cv2.getBuildInformation()
for line in info.splitlines():
    if any(k in line for k in ("GStreamer", "FFMPEG", "V4L", "CUDA")):
        print("   ", line.strip())
PY

SEP "9. Uji cv2.VideoCapture dengan pipeline yang sama"
python3 - <<'PY'
import cv2, time

def coba(nama, pipeline):
    print()
    print("---", nama, "---")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("   isOpened() = False -> pipeline tidak terbuka sama sekali")
        cap.release()
        return False
    print("   isOpened() = True")
    berhasil = 0
    t0 = time.time()
    for i in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            berhasil += 1
            if berhasil == 1:
                print("   frame pertama:", frame.shape, frame.dtype)
        else:
            print("   read() gagal pada percobaan ke", i + 1)
    dt = time.time() - t0
    cap.release()
    print(f"   {berhasil}/30 frame terbaca dalam {dt:.2f} detik")
    return berhasil > 0

base = ("nvarguscamerasrc sensor-id=0 sensor-mode={mode} ! "
        "video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
        "nvvidconv flip-method={flip} ! "
        "video/x-raw,width={ow},height={oh},format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false")

hasil = {}
hasil["mode1 1080p60 -> 1280x720"] = coba(
    "mode 1, 1920x1080 @60, skala ke 1280x720",
    base.format(mode=1, w=1920, h=1080, fps=60, flip=2, ow=1280, oh=720))

hasil["mode1 1080p60 -> 1920x1080"] = coba(
    "mode 1, tanpa penskalaan",
    base.format(mode=1, w=1920, h=1080, fps=60, flip=2, ow=1920, oh=1080))

hasil["mode0 2160p30 -> 1280x720"] = coba(
    "mode 0, 3840x2160 @30, skala ke 1280x720",
    base.format(mode=0, w=3840, h=2160, fps=30, flip=2, ow=1280, oh=720))

print()
print("=== RINGKASAN ===")
for k, v in hasil.items():
    print(f"   {'BISA  ' if v else 'GAGAL '} {k}")
PY

SEP "SELESAI"
echo "Kirimkan seluruh isi /tmp/diag_camera.log"
