#!/usr/bin/env bash
# V5 di Jetson (JetPack 6): bangun engine TensorRT dari ONNX lalu ukur latensi dan RAM.
#
# Pakai (dari root repo di Jetson; berkas .onnx disalin dari laptop ke ~/moodin_runs/v5):
#   bash scripts/validasi/v5_jetson_trt.sh ~/moodin_runs/v5
# Kalibrasi INT8 ST-GCN butuh kalib_stgcn.npy (N, 2, 90, 68) float32 di folder yang sama,
# dibuat di laptop dengan: python -m scripts.validasi.v5_kalibrasi_int8 buat ...
# Semua keluaran (engine, log, CSV) ditulis ke folder itu, tidak ke repo.
set -euo pipefail

DIR="${1:?folder keluaran V5, misalnya ~/moodin_runs/v5}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
cd "$DIR"
[ -x "$TRTEXEC" ] || { echo "trtexec tidak ditemukan di $TRTEXEC (set TRTEXEC=...)" >&2; exit 2; }

{
  echo "== versi"
  cat /etc/nv_tegra_release || true
  dpkg -l | grep -E "nvidia-jetpack|tensorrt" | awk '{print $2, $3}' || true
} | tee versi.txt

ukur() {  # $1 nama, sisanya argumen trtexec
  local nama="$1"; shift
  echo "== $nama"
  "$TRTEXEC" "$@" --iterations=200 --warmUp=2000 --avgRuns=50 --useSpinWait \
    --exportTimes="${nama}_waktu.json" 2>&1 | tee "${nama}.log" | grep -E "mean|median|percentile|Throughput" || true
}

ukur stcrppg_fp16 --onnx=stcrppg.onnx --fp16 --saveEngine=stcrppg_fp16.engine
ukur stgcn_fp16 --onnx=stgcn_va.onnx --fp16 --saveEngine=stgcn_fp16.engine

if [ -f stgcn_int8.cache ]; then
  ukur stgcn_int8 --onnx=stgcn_va.onnx --int8 --fp16 --calib=stgcn_int8.cache --saveEngine=stgcn_int8.engine
else
  echo "Lewati INT8: stgcn_int8.cache belum ada. Buat dengan:"
  echo "  python3 -m scripts.validasi.v5_kalibrasi_int8 kalibrasi --onnx $DIR/stgcn_va.onnx --data $DIR/kalib_stgcn.npy --cache $DIR/stgcn_int8.cache"
fi

echo "== RAM saat inferensi ujung ke ujung"
echo "Jalankan di terminal lain: python3 inference.py   (tutup dengan q)"
echo "Merekam tegrastats 60 detik ke tegrastats.txt ..."
timeout 60 tegrastats --interval 1000 > tegrastats.txt || true
grep -o "RAM [0-9]*/[0-9]*MB" tegrastats.txt | awk -F'[ /]' '{print $2}' | sort -n | tail -1 | xargs -I{} echo "RAM puncak (MB): {}"
echo "Selesai. Kirim balik: versi.txt, *_fp16.log, stgcn_int8.log (bila ada), tegrastats.txt, dan FPS dari overlay inference.py."
