#!/usr/bin/env bash
# Kunci kontrol kamera UVC (usulan Logitech C920) sebelum akuisisi.
#
# Pakai di Jetson:
#   bash scripts/kunci_kontrol_uvc.sh [perangkat] [berkas_konfigurasi]
#   bawaan: /dev/video0 dan config/uvc_kontrol.conf
#
# Skrip membaca daftar kontrol dengan `v4l2-ctl --list-ctrls`, lalu hanya
# mengatur kontrol yang ADA. Nama kontrol berbeda antarversi kernel, sehingga
# kedua nama ditangani (misalnya exposure_auto dan auto_exposure). Mode otomatis
# dimatikan lebih dulu, baru nilai manual diatur. Di akhir, nilai yang terbaca
# kembali dicetak. Skrip tidak merekam atau menyimpan frame.
set -euo pipefail

DEV="${1:-/dev/video0}"
CONF="${2:-$(dirname "$0")/../config/uvc_kontrol.conf}"
V4L2="${V4L2_CTL:-v4l2-ctl}"

[ -f "$CONF" ] || { echo "Berkas konfigurasi tidak ada: $CONF" >&2; exit 2; }

baca_conf() {  # baca KUNCI=nilai tanpa menjalankan isi berkas
  local kunci="$1"
  local nilai
  nilai="$(grep -E "^[[:space:]]*${kunci}=" "$CONF" | tail -n1 | cut -d= -f2- | tr -d '[:space:]')"
  [[ "$nilai" =~ ^[0-9]+$ ]] || { echo "Nilai $kunci tidak valid di $CONF: '$nilai'" >&2; exit 2; }
  echo "$nilai"
}

EXPOSURE="$(baca_conf EXPOSURE)"
WB_TEMP="$(baca_conf WB_TEMPERATURE)"
FOCUS="$(baca_conf FOCUS)"

DAFTAR="$("$V4L2" -d "$DEV" --list-ctrls)"
ada() { grep -qE "^[[:space:]]*$1[[:space:]]" <<<"$DAFTAR"; }

atur_pertama_yang_ada() {  # atur_pertama_yang_ada nilai nama1 [nama2 ...]
  local nilai="$1"; shift
  local nama
  for nama in "$@"; do
    if ada "$nama"; then
      "$V4L2" -d "$DEV" --set-ctrl="${nama}=${nilai}"
      echo "diatur: ${nama}=${nilai}"
      DIATUR+=("$nama")
      return 0
    fi
  done
  echo "tidak ada di perangkat ini: $*"
}

DIATUR=()
# 1. Frekuensi jala-jala 50 Hz (menu: 0 mati, 1 = 50 Hz, 2 = 60 Hz).
atur_pertama_yang_ada 1 power_line_frequency
# 2. Eksposur manual (menu: 1 = Manual Mode pada kedua nama).
atur_pertama_yang_ada 1 auto_exposure exposure_auto
atur_pertama_yang_ada 0 exposure_dynamic_framerate exposure_auto_priority
atur_pertama_yang_ada "$EXPOSURE" exposure_time_absolute exposure_absolute
# 3. White balance manual.
atur_pertama_yang_ada 0 white_balance_automatic white_balance_temperature_auto
atur_pertama_yang_ada "$WB_TEMP" white_balance_temperature
# 4. Fokus otomatis mati, fokus manual.
atur_pertama_yang_ada 0 focus_automatic_continuous focus_auto
atur_pertama_yang_ada "$FOCUS" focus_absolute

echo "--- nilai terbaca kembali"
for nama in "${DIATUR[@]}"; do
  "$V4L2" -d "$DEV" --get-ctrl="$nama"
done
