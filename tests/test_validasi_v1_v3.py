"""Uji sintetis skrip validasi V1 (NME 300-W) dan V3 (CHROM UBFC-Phys)."""

import numpy as np
import pytest

from data.preprocessing.chrom import chrom
from models.stc_rppg.postprocessing import rPPGProcessor
from scripts.validasi.v1_nme_300w import (
    _pilih_terdekat,
    baca_pts,
    daftar_gambar,
    ringkas,
)
from scripts.validasi.v3_chrom_ubfcphys import (
    FS_BVP,
    hr_per_jendela,
    ringkas as ringkas_v3,
)


# ---------------------------------------------------------------- V1
def _tulis_pts(path, pts):
    teks = (
        "version: 1\nn_points:  68\n{\n"
        + "\n".join(f"{x} {y}" for x, y in pts)
        + "\n}\n"
    )
    path.write_text(teks)


def test_baca_pts_1_based_ke_0_based(tmp_path):
    pts = np.column_stack([np.arange(68) + 1.0, np.arange(68) + 11.0])
    _tulis_pts(tmp_path / "a.pts", pts)
    out = baca_pts(tmp_path / "a.pts")
    assert out.shape == (68, 2) and out[0].tolist() == [0.0, 10.0]
    (tmp_path / "b.pts").write_text("{\n1 2\n}\n")
    with pytest.raises(ValueError):
        baca_pts(tmp_path / "b.pts")


def test_daftar_gambar_hanya_indoor_outdoor(tmp_path):
    for sub in ("01_Indoor", "02_Outdoor", "lain"):
        (tmp_path / "300w" / sub).mkdir(parents=True)
        _tulis_pts(tmp_path / "300w" / sub / "x_001.pts", np.zeros((68, 2)) + 1)
        (tmp_path / "300w" / sub / "x_001.png").write_bytes(b"")
    pasangan = daftar_gambar(tmp_path / "300w")
    assert sorted(p.parent.name for p, _ in pasangan) == ["01_Indoor", "02_Outdoor"]


def test_ringkasan_nme_auc_dan_gagal():
    r = ringkas([0.0, 0.04, float("nan"), 0.2], [10.0, 20.0])
    assert r["persen_gagal"] == 50.0 and r["tanpa_deteksi"] == 1
    # CED: 1/4 di 0, 2/4 dari 0,04 sampai 0,08 -> AUC = (0,25*0,04 + 0,5*0,04)/0,08 = 0,375
    assert r["auc_0_08"] == pytest.approx(0.375, abs=0.002)
    assert r["ms_per_gambar_cpu"] == 15.0


def test_pilih_wajah_terdekat_dengan_acuan():
    gt = np.array([[10.0, 10.0], [20.0, 20.0]])
    a, b = gt + 100, gt + 1
    assert _pilih_terdekat([a, b], gt) is b and _pilih_terdekat([], gt) is None


# ---------------------------------------------------------------- V3
def _rgb_sintetis(hr_bpm=72.0, fs=35.14, detik=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * detik)) / fs
    pulsa = np.sin(2 * np.pi * hr_bpm / 60 * t)
    cahaya = 1 + 0.05 * np.sin(2 * np.pi * 0.2 * t)  # perubahan intensitas (mode umum)
    kulit = np.array([180.0, 120.0, 100.0])
    arah_pulsa = np.array([0.33, 0.77, 0.53])
    rgb = cahaya[:, None] * (
        kulit[None] * (1 + 0.003 * pulsa[:, None] * arah_pulsa[None])
    )
    return rgb + rng.normal(0, 0.05, rgb.shape), fs


def test_chrom_memulihkan_denyut_sintetis():
    rgb, fs = _rgb_sintetis(72.0)
    rgb[100:105] = np.nan  # frame tanpa wajah
    s = chrom(rgb, fs)
    assert abs(rPPGProcessor(fps=fs).estimate_bpm(s) - 72.0) < 1.5


def test_chrom_menolak_masukan_tidak_valid():
    with pytest.raises(ValueError):
        chrom(np.zeros((100, 2)), 30.0)
    with pytest.raises(ValueError):
        chrom(np.ones((20, 3)), 30.0)


def test_hr_per_jendela_dan_ringkasan():
    fs = 35.0
    t_r = np.arange(int(fs * 125)) / fs
    t_b = np.arange(int(FS_BVP * 125)) / FS_BVP
    r = np.sin(2 * np.pi * 1.2 * t_r)
    b = np.sin(2 * np.pi * 1.2 * t_b)
    baris = hr_per_jendela(r, fs, b, 60.0)
    assert len(baris) == 2
    assert all(abs(x["galat"]) < 0.5 for x in baris)
    s = ringkas_v3(baris)
    assert s["n_jendela"] == 2 and s["mae_bpm"] < 0.5
