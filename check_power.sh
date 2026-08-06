#!/usr/bin/env bash
# MoodIn - diagnosa over-current pada Jetson Orin Nano Super
#
# Koreksi: perintah `grep -c 'throttl'` yang saya berikan sebelumnya SALAH.
# tegrastats tidak pernah mencetak kata "throttl", jadi hasil 0 tidak berarti
# apa-apa. Indikator yang benar adalah VDD_IN (daya masuk) dan status OC di
# sysfs. Skrip ini memakai indikator yang benar.

LOG="${1:-$HOME/moodin_logs/tegrastats.log}"

echo "===== 1. Pencacah over-current dari kernel ====="
echo "Angka yang NAIK saat inference jalan = proteksi arus benar-benar aktif."
for f in /sys/class/hwmon/hwmon*/oc*_throt_en \
         /sys/class/hwmon/hwmon*/oc_event \
         /sys/devices/platform/*/oc_event; do
  [ -e "$f" ] && echo "  $f = $(cat "$f" 2>/dev/null)"
done
if command -v nvidia-smi > /dev/null 2>&1; then :; fi
cat /sys/kernel/debug/bpmp/debug/soctherm/oc1/* 2>/dev/null | head -20

echo
echo "===== 2. Daya masuk terukur (VDD_IN) ====="
if [ -f "$LOG" ]; then
  echo "Sepuluh pembacaan terakhir:"
  grep -o 'VDD_IN [0-9]*mW' "$LOG" | tail -10
  echo
  echo "Puncak VDD_IN sepanjang log:"
  grep -o 'VDD_IN [0-9]*mW' "$LOG" | grep -o '[0-9]*' | sort -n | tail -1 \
    | awk '{printf "  %d mW  (%.1f W)\n", $1, $1/1000}'
  echo
  echo "Puncak VDD_CPU_GPU_CV:"
  grep -o 'VDD_CPU_GPU_CV [0-9]*mW' "$LOG" | grep -o '[0-9]*' | sort -n | tail -1 \
    | awk '{printf "  %d mW  (%.1f W)\n", $1, $1/1000}'
else
  echo "  Log tidak ditemukan: $LOG"
  echo "  Jalankan: sudo tegrastats --interval 1000 --logfile $LOG &"
fi

echo
echo "===== 3. Pemakaian CPU per core ====="
echo "MediaPipe berjalan di CPU (XNNPACK). Kalau semua core mendekati 100%,"
echo "itu sumber arus utamanya, bukan GPU."
if [ -f "$LOG" ]; then
  tail -3 "$LOG" | grep -o 'CPU \[[^]]*\]'
fi

echo
echo "===== 4. Interpretasi ====="
echo "  VDD_IN puncak < 15 W tapi tetap muncul Over-current"
echo "    -> bukan beban berlebih. Penyebabnya adaptor/kabel: lonjakan arus"
echo "       sesaat membuat tegangan turun di bawah ambang. Ganti ke adaptor"
echo "       barrel 19 V resmi, atau kabel USB-C yang lebih pendek/tebal."
echo "  VDD_IN puncak mendekati atau melebihi batas mode (15/25 W)"
echo "    -> beban memang berlebih. Turunkan proc_width/proc_height di"
echo "       config/default.yaml, atau pakai mode 15W."
