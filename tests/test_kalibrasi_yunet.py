"""Kalibrasi kotak YuNet ke gaya kotak HOG dlib (T1.2): pemasangan, penerapan, dan pemisahan data."""

import json

import numpy as np
import pytest

from scripts.validasi import v1_kalibrasi_yunet as K


def _kotak_acak(rng, n):
    out = []
    for _ in range(n):
        x, y = rng.uniform(50, 400, 2)
        w = rng.uniform(80, 250)
        h = w * rng.uniform(1.1, 1.3)
        out.append((x, y, x + w, y + h))
    return out


def test_pasang_memulihkan_parameter_yang_diketahui():
    rng = np.random.default_rng(0)
    benar = {"dx": 0.02, "dy": 0.12, "sw": 0.9, "sh": 0.75}
    yu = _kotak_acak(rng, 200)
    pasangan = [(b, K.terapkan(b, benar)) for b in yu]
    # Sepuluh persen pasangan rusak: median harus tetap dekat parameter benar.
    for i in range(0, 200, 10):
        pasangan[i] = (pasangan[i][0], (0.0, 0.0, 10.0, 10.0))
    hasil = K.pasang(pasangan)
    for k, v in benar.items():
        assert abs(hasil[k] - v) < 0.02, (k, hasil[k])
    assert hasil["n_pasangan"] == 200
    assert hasil["sw_iqr"][0] <= hasil["sw"] <= hasil["sw_iqr"][1]


def test_terapkan_identitas_tidak_mengubah_kotak():
    b = (10.0, 20.0, 110.0, 150.0)
    assert np.allclose(K.terapkan(b, {"dx": 0, "dy": 0, "sw": 1, "sh": 1}), b)


def test_pasang_menolak_pasangan_terlalu_sedikit():
    with pytest.raises(ValueError, match="minimal 20"):
        K.pasang([((0, 0, 10, 10), (0, 0, 10, 10))] * 5)


def test_kotak_cocok_memakai_iou_minimum():
    gt = (100.0, 100.0, 200.0, 220.0)
    jauh = (400.0, 400.0, 500.0, 520.0)
    dekat = (105.0, 110.0, 205.0, 225.0)
    assert K.kotak_cocok([jauh, dekat], gt) == dekat
    assert K.kotak_cocok([jauh], gt) is None


def test_daftar_latih_menolak_folder_uji(tmp_path):
    for sub in (
        "lfpw/trainset",
        "lfpw/testset",
        "afw",
        "300w/01_Indoor",
        "300w/02_Outdoor",
    ):
        d = tmp_path / sub
        d.mkdir(parents=True)
        (d / "a.pts").write_text("x")
        (d / "a.jpg").write_bytes(b"")
    nama = sorted(
        str(p.parent.relative_to(tmp_path)).replace("\\", "/")
        for p, _ in K.daftar_latih(tmp_path)
    )
    assert nama == ["afw", "lfpw/trainset"]


def test_baca_kalibrasi_menolak_kunci_hilang(tmp_path):
    p = tmp_path / "k.json"
    p.write_text(json.dumps({"dx": 0.0, "dy": 0.1}))
    with pytest.raises(ValueError, match="sw"):
        K.baca_kalibrasi(p)
    p.write_text(json.dumps({"dx": 0.0, "dy": 0.1, "sw": 1.0, "sh": 0.8}))
    assert K.baca_kalibrasi(p)["sh"] == 0.8
