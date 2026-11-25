"""Fitur HRV pada sinyal pulsa sintetis dengan IBI yang diketahui."""

import numpy as np
import pytest

from models.stress import hrv as H

FS = 64.0


def pulsa(
    ibi_ms: np.ndarray, fs: float = FS, derau: float = 0.0, seed: int = 0
) -> np.ndarray:
    """Gelombang pulsa: satu puncak Gauss sempit per detak pada waktu kumulatif IBI."""
    t_puncak = np.cumsum(np.r_[0.5, ibi_ms / 1000.0])
    n = int((t_puncak[-1] + 0.5) * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    for tp in t_puncak:
        x += np.exp(-0.5 * ((t - tp) / 0.06) ** 2)
    rng = np.random.default_rng(seed)
    return x + derau * rng.standard_normal(n)


def ibi_bermodulasi(durasi_s, mean_ms=800.0, a_lf=0.0, a_hf=0.0, f_lf=0.1, f_hf=0.25):
    out, t = [], 0.0
    while t < durasi_s:
        v = (
            mean_ms
            + a_lf * np.sin(2 * np.pi * f_lf * t)
            + a_hf * np.sin(2 * np.pi * f_hf * t)
        )
        out.append(v)
        t += v / 1000.0
    return np.array(out)


def test_fitur_waktu_ibi_tetap():
    x = pulsa(np.full(80, 750.0))
    f = H.fitur_jendela(x, FS, H.SetelanHRV())
    assert abs(f["mean_nn"] - 750.0) < 10
    assert abs(f["mean_hr"] - 80.0) < 1.5
    assert f["rmssd"] < 20 and f["sdnn"] < 20


def test_rmssd_dan_sdnn_naik_dengan_variabilitas():
    tenang = H.fitur_jendela(pulsa(ibi_bermodulasi(60, a_hf=10)), FS, H.SetelanHRV())
    variatif = H.fitur_jendela(pulsa(ibi_bermodulasi(60, a_hf=60)), FS, H.SetelanHRV())
    assert variatif["rmssd"] > 2 * tenang["rmssd"]
    assert variatif["sdnn"] > 2 * tenang["sdnn"]


def test_saring_ibi_membuang_puncak_salah():
    ibi = np.full(30, 800.0)
    ibi[10] = 400.0  # puncak ganda
    ibi[20] = 1600.0  # puncak terlewat
    ibi[25] = 2500.0  # di luar rentang fisiologis
    t = np.cumsum(ibi) / 1000.0
    _, ibi2 = H.saring_ibi(t, ibi, H.SetelanHRV())
    assert ibi2.size == 27 and np.all(ibi2 == 800.0)


@pytest.mark.parametrize("metode", ["welch", "lombscargle"])
def test_lf_hf_mengikuti_modulasi(metode):
    s = H.SetelanHRV(metode_frekuensi=metode, fitur=H.FITUR_WAKTU + H.FITUR_FREKUENSI)
    dominan_lf = H.fitur_jendela(pulsa(ibi_bermodulasi(130, a_lf=50, a_hf=10)), FS, s)
    dominan_hf = H.fitur_jendela(pulsa(ibi_bermodulasi(130, a_lf=10, a_hf=50)), FS, s)
    assert dominan_lf["lf_hf"] > 3.0
    assert dominan_hf["lf_hf"] < 1 / 3.0


def test_frekuensi_ditolak_pada_jendela_pendek():
    s = H.SetelanHRV(fitur=H.FITUR_WAKTU + H.FITUR_FREKUENSI)
    with pytest.raises(ValueError, match="minimal 120"):
        H.fitur_jendela(pulsa(np.full(70, 800.0)), FS, s)
    s60 = H.SetelanHRV(min_frekuensi_s=60.0, fitur=H.FITUR_WAKTU + H.FITUR_FREKUENSI)
    assert "lf_hf" in H.fitur_jendela(pulsa(ibi_bermodulasi(62, a_lf=40)), FS, s60)


def test_sinyal_datar_memberi_none():
    datar = np.linspace(0, 1e-9, int(60 * FS))
    assert H.fitur_jendela(datar, FS, H.SetelanHRV()) is None


def test_setelan_menolak_isian_salah():
    with pytest.raises(ValueError):
        H.SetelanHRV(metode_frekuensi="fft")
    with pytest.raises(ValueError):
        H.SetelanHRV(fitur=("pnn50",))
