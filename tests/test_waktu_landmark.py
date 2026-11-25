"""Logika pelacakan antarframe di scripts/validasi/waktu_landmark.py, tanpa dlib."""

import numpy as np

from scripts.validasi import waktu_landmark as W


def _wajah(cx, cy, skala=1.0):
    rng = np.random.default_rng(1)
    dasar = rng.uniform(-50, 50, (68, 2))
    dasar[36], dasar[45] = (-30, 0), (30, 0)
    return dasar * skala + np.array([cx, cy])


def test_geser_kotak_mengikuti_pusat_dan_skala():
    lama, baru = _wajah(100, 100), _wajah(120, 90, 1.5)
    k = W.geser_kotak((50.0, 40.0, 150.0, 160.0), lama, baru)
    harap = np.array([100.0, 100.0]) + baru.mean(0) - lama.mean(0)
    assert np.allclose(((k[0] + k[2]) / 2, (k[1] + k[3]) / 2), harap)
    assert np.isclose(k[2] - k[0], 150.0) and np.isclose(k[3] - k[1], 180.0)


def test_pelacak_mendeteksi_tiap_n_frame():
    n_det = []
    posisi = iter(range(0, 1000, 2))

    def deteksi(_f):
        n_det.append(1)
        return (0.0, 0.0, 100.0, 120.0)

    def bentuk(_f, _k):
        return _wajah(next(posisi), 50)

    p = W.Pelacak(deteksi, bentuk, tiap=5)
    hasil = [p.langkah(None) for _ in range(12)]
    assert [ulang for _, ulang in hasil] == [True, False, False, False, False] * 2 + [
        True,
        False,
    ]
    assert len(n_det) == 3 and p.n_deteksi == 3
    assert all(lm is not None for lm, _ in hasil)


def test_pelacak_mendeteksi_ulang_bila_skala_melompat():
    skala = iter([1.0, 1.0, 2.0, 1.0, 1.0])
    p = W.Pelacak(
        lambda f: (0.0, 0.0, 100.0, 100.0),
        lambda f, k: _wajah(50, 50, next(skala)),
        tiap=50,
    )
    lm1, u1 = p.langkah(None)
    lm2, u2 = p.langkah(None)
    lm3, u3 = p.langkah(None)
    lm4, u4 = p.langkah(None)
    assert (u1, u2, u3, u4) == (True, False, False, True)
    assert lm3 is None and lm4 is not None


def test_pelacak_tanpa_wajah():
    p = W.Pelacak(lambda f: None, lambda f, k: _wajah(0, 0), tiap=3)
    assert p.langkah(None) == (None, True)
    assert p.langkah(None) == (None, True)


def test_ringkas_ms():
    r = W.ringkas_ms([10.0, 20.0, 30.0])
    assert r["n"] == 3 and r["median"] == 20.0 and r["fps_setara"] == 50.0
    assert W.ringkas_ms([]) == {"n": 0}
