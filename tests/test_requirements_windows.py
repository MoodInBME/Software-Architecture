"""Berkas requirements Windows: torch tidak dikunci, satu OpenCV, numpy dan mediapipe terkunci."""

import re
from pathlib import Path

from scripts import cek_lingkungan

REQ = Path(__file__).resolve().parents[1] / "requirements"


def _paket(nama):
    out = []
    for ln in (REQ / nama).read_text(encoding="utf-8").splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            out.append(re.split(r"[=<>!~\[]", ln, maxsplit=1)[0].strip().lower())
    return out


def _kunci():
    d = {}
    for ln in (REQ / "constraints.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            n, v = ln.split("==")
            d[n.lower()] = v
    return d


def test_torch_tidak_dikunci_dan_tidak_didaftar():
    for f in ("windows-gpu.txt", "windows-nodeps.txt", "constraints.txt"):
        assert not {"torch", "torchvision"} & set(_paket(f)), f


def test_satu_opencv_dan_kunci_utama():
    semua = _paket("windows-gpu.txt") + _paket("windows-nodeps.txt")
    assert [p for p in semua if p.startswith("opencv")] == ["opencv-contrib-python"]
    k = _kunci()
    assert k["numpy"] == "1.26.4" and k["mediapipe"] == "0.10.14"
    assert k["sympy"] == "1.13.1"  # diminta torch 2.5.1
    assert set(_paket("windows-gpu.txt")) <= set(k), (
        "setiap paket harus punya versi kunci"
    )


def test_mediapipe_hanya_di_berkas_nodeps():
    assert "mediapipe" not in _paket("windows-gpu.txt")
    assert _paket("windows-nodeps.txt") == ["mediapipe"]


def test_cek_lingkungan_berjalan_tanpa_kamera():
    masalah = cek_lingkungan.periksa()
    assert isinstance(masalah, list)
