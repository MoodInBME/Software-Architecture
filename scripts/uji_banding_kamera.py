"""Uji banding kamera: hanya menulis angka ke CSV, tidak pernah menyimpan frame.

Mengukur selang antar-frame dari cap waktu (rerata dan simpangan baku), jumlah frame
hilang, serta rerata kecerahan (luma Y) ROI per jendela 10 detik. Dipakai untuk
membandingkan backend ``csi`` (IMX477) dan ``uvc`` (C920) pada setelan yang sama.

Contoh di Jetson, dari root repo:
    python scripts/uji_banding_kamera.py --backend uvc --detik 60 \
        --keluaran ~/hasil_uji_kamera/uvc_ruangA
Keluaran: <prefix>_ringkasan.csv dan <prefix>_jendela.csv. Simpan di luar repo.

Cap waktu: bila backend memberi cap waktu driver (CAP_PROP_POS_MSEC > 0), itu yang
dipakai; selain itu ``time.perf_counter`` host setelah read() (resolusi di bawah
1 mikrodetik, juga di Windows, tidak seperti ``time.monotonic``). Kolom ``sumber_waktu``
mencatat mana yang dipakai.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.acquisition.frame_stats import FrameStats, roi_luma  # noqa: E402

ROI_BAWAAN = (
    0.3,
    0.2,
    0.4,
    0.5,
)  # pusat frame; posisi wajah saat duduk di depan cermin


def ukur(frames, expected_fps, roi=ROI_BAWAAN, window_s=10.0):
    """frames: iterator (cap_waktu_driver_s atau None, frame_bgr). Return (FrameStats, sumber)."""
    st = FrameStats(expected_fps=expected_fps, window_s=window_s)
    sumber = None
    for ts_driver, frame in frames:
        if sumber is None:
            sumber = "driver" if ts_driver is not None else "host"
        ts = ts_driver if sumber == "driver" else time.perf_counter()
        if ts is None:
            raise ValueError("Cap waktu driver hilang di tengah sesi")
        st.add(ts, roi_luma(frame, roi))
    return st, sumber or "host"


def _frames_dari_capture(cap, detik):
    import cv2

    batas = time.monotonic() + detik
    while time.monotonic() < batas:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        yield (ms / 1000.0 if ms and ms > 0 else None), frame
        del frame


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=["csi", "uvc"], required=True)
    ap.add_argument("--detik", type=float, default=60.0)
    ap.add_argument("--keluaran", required=True, help="prefix berkas CSV, di luar repo")
    ap.add_argument(
        "--roi", type=float, nargs=4, default=ROI_BAWAAN, metavar=("X", "Y", "W", "H")
    )
    ap.add_argument("--config", default=str(_ROOT / "config" / "default.yaml"))
    args = ap.parse_args()

    prefix = Path(os.path.expanduser(args.keluaran)).resolve()
    if prefix.is_relative_to(_ROOT.resolve()):
        raise SystemExit(
            "--keluaran berada di dalam repo. Simpan hasil uji di luar repo."
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import yaml

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if args.backend == "uvc":
        from data.acquisition.camera import UvcSetting, open_uvc_capture

        setting = UvcSetting.from_config(cfg)
        cap, laporan = open_uvc_capture(setting)
        print("[uvc] diminta", setting, "| dipakai driver", laporan)
        if not laporan["cocok"]:
            print("[uvc] PERINGATAN: driver tidak memakai setelan yang diminta.")
        fps = setting.fps
    else:
        import inference  # memuat konfigurasi dan pembangun pipeline CSI yang sama

        cap = cv2.VideoCapture(inference.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        fps = int(cfg.get("csi_fps", 30))
    if not cap.isOpened():
        raise SystemExit("Kamera tidak dapat dibuka.")
    try:
        st, sumber = ukur(_frames_dari_capture(cap, args.detik), fps, tuple(args.roi))
    finally:
        cap.release()
    meta = {"backend": args.backend, "fps_diminta": fps, "sumber_waktu": sumber}
    st.write_csv(f"{prefix}_ringkasan.csv", f"{prefix}_jendela.csv", meta)
    print(st.summary())


if __name__ == "__main__":
    main()
