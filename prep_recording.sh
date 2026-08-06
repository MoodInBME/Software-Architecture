#!/usr/bin/env bash
# MoodIn - persiapan sesi perekaman video demo di Jetson Orin Nano Super
#
#   ./prep_recording.sh            -> siapkan sesi
#   ./prep_recording.sh --restore  -> kembalikan setelan semula
#
# Tujuan: mematikan semua yang bisa memunculkan popup di atas jendela
# inferensi, dan menstabilkan daya supaya tidak ada throttling di tengah
# pengambilan gambar.

set -u
MODE="${1:-setup}"
LOGDIR="$HOME/moodin_logs"

if [ "$MODE" = "--restore" ]; then
  echo "[1/4] Mengaktifkan kembali notifikasi..."
  gsettings set org.gnome.desktop.notifications show-banners true
  echo "[2/4] Mengaktifkan kembali blanking layar..."
  gsettings set org.gnome.desktop.session idle-delay 300
  gsettings set org.gnome.desktop.screensaver lock-enabled true
  echo "[3/4] Menghentikan pencatatan tegrastats..."
  pkill -f "tegrastats --logfile" 2>/dev/null || true
  echo "[4/4] Selesai. Mode daya TIDAK dikembalikan otomatis."
  echo "      Cek dengan: sudo nvpmodel -q"
  exit 0
fi

mkdir -p "$LOGDIR"

echo "===== Mode daya saat ini ====="
sudo nvpmodel -q || true
echo
echo "Mode yang tersedia di papan ini:"
grep -E "^< POWER_MODEL" /etc/nvpmodel.conf 2>/dev/null || \
  echo "  (tidak terbaca - cek manual: cat /etc/nvpmodel.conf)"
echo
echo "CATATAN: MAXN SUPER tidak punya batas daya, jadi papan boleh menarik"
echo "arus melebihi kemampuan adaptor -> muncul 'Over-current'."
echo "Untuk perekaman, mode berbatas jauh lebih aman daripada mode tercepat"
echo "yang terus-menerus di-throttle."
echo
read -r -p "Pindah ke mode 25W sekarang? (nomor mode, kosongkan untuk lewati): " PM
if [ -n "$PM" ]; then
  sudo nvpmodel -m "$PM"
  echo "Mode diubah. Verifikasi:"
  sudo nvpmodel -q
fi

echo
echo "[1/4] Mematikan banner notifikasi GNOME..."
gsettings set org.gnome.desktop.notifications show-banners false

echo "[2/4] Mematikan blanking dan lock layar..."
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false

echo "[3/4] Menyembunyikan dock (jika ekstensi dash-to-dock aktif)..."
gsettings set org.gnome.shell.extensions.dash-to-dock autohide true 2>/dev/null || true
gsettings set org.gnome.shell.extensions.dash-to-dock intellihide true 2>/dev/null || true

echo "[4/4] Mulai mencatat tegrastats ke $LOGDIR/tegrastats.log ..."
pkill -f "tegrastats --logfile" 2>/dev/null || true
sudo tegrastats --interval 1000 --logfile "$LOGDIR/tegrastats.log" &
sleep 1

echo
echo "Sesi laptop siap. Jalankan inference.py sekarang."
echo
echo "Setelah selesai merekam, periksa apakah sempat throttling:"
echo "  grep -c 'throttl' $LOGDIR/tegrastats.log"
echo "  grep -o 'GR3D_FREQ [0-9]*%' $LOGDIR/tegrastats.log | tail -20"
echo "  grep -o 'tj@[0-9.]*C' $LOGDIR/tegrastats.log | tail -20"
echo
echo "Lalu kembalikan setelan dengan: ./prep_recording.sh --restore"
