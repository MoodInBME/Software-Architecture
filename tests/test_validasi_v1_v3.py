"""Uji sintetis skrip validasi V1 (NME 300-W) dan V3 (CHROM UBFC-Phys)."""

import numpy as np
import pytest

from data.preprocessing.chrom import chrom
from models.stc_rppg.postprocessing import rPPGProcessor
from scripts.validasi.v1_nme_300w import (
    _pilih_terdekat,
    baca_pts,
    daftar_gambar,
    iou,
    kotak,
    nilai_gambar,
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


def _wajah(dx=0.0, skala=1.0):
    t = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    p = np.column_stack([100 + 40 * np.cos(t), 100 + 50 * np.sin(t)])
    p[36], p[45] = [80.0, 90.0], [120.0, 90.0]  # jarak mata luar 40 piksel
    return (p - 100) * skala + 100 + np.array([dx, 0.0])


def test_nilai_gambar_per_wilayah_dan_cocok():
    gt = _wajah()
    pred = gt.copy()
    pred[0:17] += np.array([4.0, 0.0])  # hanya rahang bergeser 4 piksel = 0,1 jarak mata
    r = nilai_gambar(pred, gt)
    assert r["cocok"] and r["nme_rahang"] == pytest.approx(0.1) and r["nme_mata"] == pytest.approx(0.0)
    assert r["nme"] == pytest.approx(0.1 * 17 / 68)
    salah = nilai_gambar(_wajah(dx=500.0), gt)  # wajah lain di gambar yang sama
    assert salah["terdeteksi"] and not salah["cocok"]
    kosong = nilai_gambar(None, gt)
    assert not kosong["terdeteksi"] and np.isnan(kosong["nme"])
    assert iou(kotak(gt), kotak(gt)) == pytest.approx(1.0)


def test_ringkasan_hanya_menghitung_nme_pada_wajah_cocok():
    gt = _wajah()
    baris = [
        nilai_gambar(gt, gt),  # sempurna
        nilai_gambar(gt + np.array([1.6, 0.0]), gt),  # NME 0,04
        nilai_gambar(None, gt),  # tidak terdeteksi
        nilai_gambar(_wajah(dx=500.0), gt),  # wajah salah: tidak masuk rerata NME
    ]
    r = ringkas(baris, [10.0, 20.0])
    assert r["persen_terdeteksi"] == 75.0 and r["persen_cocok"] == 50.0
    assert r["nme_rerata_cocok"] == pytest.approx(0.02)
    assert r["persen_gagal"] == 50.0
    # CED: 1/4 di 0, 2/4 dari 0,04 sampai 0,08 -> AUC = (0,25*0,04 + 0,5*0,04)/0,08 = 0,375
    assert r["auc_0_08"] == pytest.approx(0.375, abs=0.002)
    assert set(r["per_wilayah_cocok"]) == {"rahang", "alis", "hidung", "mata", "mulut"}
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
