"""Uji abstraksi kamera, alat uji banding, dan skrip kunci kontrol UVC (frame sintetis).

Jalankan dari root repo dengan Python 3.12:

    python -m pytest tests/test_camera_acquisition.py -q
"""

import csv
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from data.acquisition.camera import UvcSetting, camera_backend, open_uvc_capture
from data.acquisition.frame_stats import FrameStats, roi_luma
from scripts.uji_banding_kamera import ukur

REPO = Path(__file__).resolve().parents[1]
SKRIP = REPO / "scripts" / "kunci_kontrol_uvc.sh"


def _frame(b, g, r, h=48, w=64):
    f = np.zeros((h, w, 3), np.uint8)
    f[..., 0], f[..., 1], f[..., 2] = b, g, r
    return f


def test_roi_luma_bt601():
    f = _frame(10, 100, 200)
    assert roi_luma(f, (0.0, 0.0, 1.0, 1.0)) == pytest.approx(
        0.114 * 10 + 0.587 * 100 + 0.299 * 200
    )
    f[:24] = 0
    assert (
        roi_luma(f, (0.0, 0.5, 1.0, 0.5)) > 0 and roi_luma(f, (0.0, 0.0, 1.0, 0.5)) == 0
    )
    with pytest.raises(ValueError):
        roi_luma(f, (1.0, 1.0, 0.1, 0.1))


def test_selang_dan_frame_hilang():
    fps = 30.0
    t = np.arange(0, 20 * fps) / fps
    t = np.delete(t, [100, 200, 201])  # 3 frame hilang
    st = FrameStats(expected_fps=fps)
    for k, ts in enumerate(t):
        st.add(float(ts), 50.0 if ts < 10 else 80.0)
    s = st.summary()
    assert s["frame_hilang"] == 3
    assert s["frames"] == len(t)
    assert s["selang_rerata_ms"] == pytest.approx(
        1000 * (t[-1] - t[0]) / (len(t) - 1), abs=1e-3
    )
    w = st.windows()
    assert [x["luma_rerata"] for x in w] == [50.0, 80.0]
    with pytest.raises(ValueError):
        st.add(1.0, 10.0)  # cap waktu mundur


def test_csv_hanya_angka(tmp_path):
    st = FrameStats(expected_fps=30.0)
    for k in range(90):
        st.add(k / 30.0, 60.0)
    st.write_csv(str(tmp_path / "r.csv"), str(tmp_path / "j.csv"), {"backend": "uvc"})
    with open(tmp_path / "r.csv", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["backend"] == "uvc" and row["frame_hilang"] == "0"
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".csv", ".csv"]


def test_ukur_sumber_waktu():
    frames = [(k / 30.0, _frame(0, 128, 0)) for k in range(30)]
    st, sumber = ukur(iter(frames), 30.0)
    assert sumber == "driver" and st.summary()["frames"] == 30
    st, sumber = ukur(iter([(None, _frame(0, 0, 0)) for _ in range(3)]), 30.0)
    assert sumber == "host" and len(st.t) == 3


def test_backend_bawaan_dan_validasi():
    assert camera_backend({}) == "csi"
    assert camera_backend({"camera_backend": "UVC"}) == "uvc"
    with pytest.raises(ValueError):
        camera_backend({"camera_backend": "rtsp"})


class _CapPalsu:
    def __init__(self, cv2, src, api, hasil):
        self.cv2, self.src, self.api, self.hasil, self.set_calls = (
            cv2,
            src,
            api,
            hasil,
            [],
        )

    def set(self, prop, val):
        self.set_calls.append((prop, val))

    def get(self, prop):
        return self.hasil[prop]


class _Cv2Palsu:
    (
        CAP_V4L2,
        CAP_PROP_FOURCC,
        CAP_PROP_FRAME_WIDTH,
        CAP_PROP_FRAME_HEIGHT,
        CAP_PROP_FPS,
    ) = 200, 6, 3, 4, 5

    def __init__(self, fourcc="YUYV", w=640, h=480, fps=30.0):
        code = sum(ord(c) << (8 * i) for i, c in enumerate(fourcc))
        self.hasil = {6: code, 3: w, 4: h, 5: fps}

    @staticmethod
    def VideoWriter_fourcc(*c):
        return sum(ord(x) << (8 * i) for i, x in enumerate(c))

    def VideoCapture(self, src, api):
        self.cap = _CapPalsu(self, src, api, self.hasil)
        return self.cap


def test_open_uvc_meminta_yuyv_v4l2():
    cv2 = _Cv2Palsu()
    cap, lap = open_uvc_capture(UvcSetting(device="/dev/video2"), cv2_module=cv2)
    assert cap.api == cv2.CAP_V4L2 and cap.src == "/dev/video2"
    assert (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV")) in cap.set_calls
    assert lap["cocok"] and lap["fourcc"] == "YUYV"
    _, lap = open_uvc_capture(
        UvcSetting(device="0"), cv2_module=_Cv2Palsu(fourcc="MJPG")
    )
    assert not lap["cocok"]


DAFTAR_BARU = """\
                     brightness 0x00980900 (int)    : min=0 max=255 step=1 default=128 value=128
          power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=2
      white_balance_automatic 0x0098090c (bool)   : default=1 value=1
    white_balance_temperature 0x0098091a (int)    : min=2000 max=6500 step=1 default=4000 value=4000 flags=inactive
                auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
       exposure_time_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=250 value=250 flags=inactive
   exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1
               focus_absolute 0x009a090a (int)    : min=0 max=250 step=5 default=0 value=0 flags=inactive
   focus_automatic_continuous 0x009a090c (bool)   : default=1 value=1
"""
DAFTAR_LAMA = """\
          power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=2
white_balance_temperature_auto 0x0098090c (bool)   : default=1 value=1
    white_balance_temperature 0x0098091a (int)    : min=2000 max=6500 step=1 default=4000 value=4000
                exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
            exposure_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=250 value=250
       exposure_auto_priority 0x009a0903 (bool)   : default=0 value=1
                   focus_auto 0x009a090c (bool)   : default=1 value=1
"""


# Skrip kunci_kontrol_uvc.sh ditujukan untuk Jetson/Linux (v4l2-ctl). Di Windows, bash
# yang tersedia (Git Bash atau WSL) memetakan jalur dan izin eksekusi stub v4l2-ctl
# secara berbeda, sehingga hasil uji tidak mewakili perangkat sasaran.
LEWATI_DI_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32",
    reason="skrip V4L2 untuk Jetson/Linux; bash di Windows (Git Bash/WSL) tidak mewakili sasaran",
)


def _jalankan(tmp_path, daftar):
    stub = tmp_path / "v4l2-ctl"
    log = tmp_path / "panggilan.log"
    (tmp_path / "daftar.txt").write_text(daftar)
    stub.write_text(
        textwrap.dedent(f"""\
        #!/usr/bin/env bash
        echo "$*" >> "{log}"
        for a in "$@"; do
          case "$a" in --list-ctrls) cat "{tmp_path / "daftar.txt"}";; --get-ctrl=*) echo "${{a#--get-ctrl=}}: ok";; esac
        done
        """)
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    conf = tmp_path / "uvc.conf"
    conf.write_text("EXPOSURE=200\nWB_TEMPERATURE=4500\nFOCUS=30\n")
    env = dict(os.environ, V4L2_CTL=str(stub))
    r = subprocess.run(
        ["bash", str(SKRIP), "/dev/video9", str(conf)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    sets = [
        ln.split("--set-ctrl=")[1]
        for ln in log.read_text().splitlines()
        if "--set-ctrl=" in ln
    ]
    return sets, r.stdout


@LEWATI_DI_WINDOWS
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash tidak ada")
def test_skrip_kunci_nama_kernel_baru(tmp_path):
    sets, _ = _jalankan(tmp_path, DAFTAR_BARU)
    assert sets == [
        "power_line_frequency=1",
        "auto_exposure=1",
        "exposure_dynamic_framerate=0",
        "exposure_time_absolute=200",
        "white_balance_automatic=0",
        "white_balance_temperature=4500",
        "focus_automatic_continuous=0",
        "focus_absolute=30",
    ]


@LEWATI_DI_WINDOWS
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash tidak ada")
def test_skrip_kunci_nama_kernel_lama_dan_kontrol_hilang(tmp_path):
    sets, out = _jalankan(tmp_path, DAFTAR_LAMA)
    assert sets == [
        "power_line_frequency=1",
        "exposure_auto=1",
        "exposure_auto_priority=0",
        "exposure_absolute=200",
        "white_balance_temperature_auto=0",
        "white_balance_temperature=4500",
        "focus_auto=0",
    ]
    assert "tidak ada di perangkat ini: focus_absolute" in out


@LEWATI_DI_WINDOWS
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash tidak ada")
def test_skrip_menolak_nilai_tidak_valid(tmp_path):
    conf = tmp_path / "uvc.conf"
    conf.write_text("EXPOSURE=$(reboot)\nWB_TEMPERATURE=4500\nFOCUS=30\n")
    r = subprocess.run(
        ["bash", str(SKRIP), "/dev/video9", str(conf)],
        env=dict(os.environ, V4L2_CTL="/bin/false"),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2 and "tidak valid" in r.stderr
