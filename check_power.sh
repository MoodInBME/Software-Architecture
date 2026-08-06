#!/usr/bin/env bash
# MoodIn - diagnosa over-current pada Jetson Orin Nano Super  (v2)
#
# KOREKSI TERHADAP v1:
#   - v1 membaca oc*_throt_en dan menyebutnya "pencacah". Itu SALAH.
#     oc*_throt_en hanya menandakan proteksi OC AKTIF (selalu 1), bukan riwayat.
#   - v1 menebak jalur /sys/class/hwmon/hwmon4/in1_input. Jalur hwmon tidak
#     stabil antar boot. v2 menemukannya sendiri lewat file 'name'.
#
# Riwayat over-current yang sebenarnya hanya ada di dmesg. Kalau dmesg bersih,
# tidak ada throttling yang benar-benar terjadi.

LOG="${1:-$HOME/moodin_logs/tegrastats.log}"

echo "===== 1. Uptime ====="
echo "  $(uptime -p)   (dmesg hanya mencakup rentang ini)"

echo
echo "===== 2. Kejadian over-current NYATA dari kernel ====="
OC=$(sudo dmesg 2>/dev/null | grep -i -E 'over.?current|oc[0-9]_alarm|oc_event|soctherm.*alarm' \
     | grep -v 'cooling device registered')
if [ -z "$OC" ]; then
  echo "  TIDAK ADA. Kernel tidak pernah mencatat kejadian over-current."
  echo "  Baris 'thermal-trip-event ... cooling device registered' pada boot"
  echo "  BUKAN kejadian throttling, itu hanya pendaftaran driver."
else
  echo "$OC"
fi

echo
echo "===== 3. Peta sensor hwmon (jalur berubah tiap boot) ====="
for d in /sys/class/hwmon/hwmon*; do
  n=$(cat "$d/name" 2>/dev/null)
  echo "  $d  name=$n"
done

echo
echo "===== 4. Tegangan dan arus masuk terukur ====="
FOUND=0
for f in /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_input \
         /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/curr*_input; do
  if [ -e "$f" ]; then
    FOUND=1
    lbl_file="${f%_input}_label"
    lbl=$(cat "$lbl_file" 2>/dev/null)
    echo "  $(basename "$f")  ${lbl}  = $(cat "$f" 2>/dev/null)"
  fi
done
if [ "$FOUND" = "0" ]; then
  echo "  Driver ina3221 tidak ditemukan pada jalur baku."
  echo "  Coba manual:"
  echo "    sudo find /sys -name 'in1_input' 2>/dev/null | head"
fi
echo
echo "  Satuan: in*_input dalam mV, curr*_input dalam mA."
echo "  in1 mendekati 19000 mV -> barrel jack 19 V, arus rendah, aman."
echo "  in1 mendekati  5000 mV -> USB-C 5 V, 18 W berarti 3,6 A, MELEBIHI 3 A."

echo
echo "===== 5. Daya masuk dari log tegrastats ====="
if [ -f "$LOG" ]; then
  echo "  Puncak VDD_IN:"
  grep -o 'VDD_IN [0-9]*mW' "$LOG" | grep -o '[0-9]*' | sort -n | tail -1 \
    | awk '{printf "    %d mW  (%.1f W)\n", $1, $1/1000}'
  echo "  Puncak VDD_CPU_GPU_CV:"
  grep -o 'VDD_CPU_GPU_CV [0-9]*mW' "$LOG" | grep -o '[0-9]*' | sort -n | tail -1 \
    | awk '{printf "    %d mW  (%.1f W)\n", $1, $1/1000}'
  echo "  CPU tiga pembacaan terakhir:"
  tail -3 "$LOG" | grep -o 'CPU \[[^]]*\]' | sed 's/^/    /'
else
  echo "  Log tidak ada: $LOG"
  echo "  Jalankan dulu: ./prep_recording.sh"
fi

echo
echo "===== 6. Kesimpulan ====="
echo "  Bagian 2 kosong + suhu di bawah 60 C + GR3D_FREQ rendah"
echo "    -> tidak ada throttling nyata. Notifikasi desktop bersifat kosmetik"
echo "       dan cukup dibungkam lewat ./prep_recording.sh sebelum merekam."
echo "  Bagian 2 berisi baris over-current saat inference jalan"
echo "    -> nyata. Periksa tegangan di bagian 4 sebelum menyalahkan beban."
