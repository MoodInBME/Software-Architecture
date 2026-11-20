"""Uji B.1b: decision_margin (Chebyshev), mode evaluasi dan tampilan, serta late_fusion.

Jalankan dari root repo dengan Python 3.12:

    python -m pytest tests/test_mood_margin.py -q
"""

import itertools
import random

import numpy as np
import pytest

from evaluation.va_metrics import evaluate_va
from models.fusion.late_fusion import LateFusionVABridge
from models.st_gcn_graph_tran.labels import MOOD_TO_ID
from models.st_gcn_graph_tran.labels_tabel33 import sam_to_unit
from models.st_gcn_graph_tran.mood_rules import (
    MOOD_GAP_ID,
    MoodDisplayHysteresis,
    boundary_distance,
    decision_margin,
    mood_id_function,
    va_to_mood_prediction,
)

PUSAT = {
    "Netral": (0.0, 0.0),
    "Ceria": (0.6, 0.6),
    "Rileks": (0.6, -0.6),
    "Cemas": (-0.6, 0.6),
    "Tertekan": (-0.6, -0.6),
}
BATAS = [
    (0.25, 0.1),
    (0.25, 0.25),
    (0.0, 0.5),
    (0.5, 0.0),
    (-0.25, -0.1),
    (0.1, -0.25),
    (0.0, -0.9),
]


def _tabel33_acuan(v_sam, a_sam):
    """Tabel 3.3 ditulis ulang langsung dari rating SAM, terpisah dari kode yang diuji."""
    if 4 <= v_sam <= 6 and 4 <= a_sam <= 6:
        return "Netral"
    if v_sam > 5:
        return "Ceria" if a_sam > 5 else "Rileks"
    if v_sam < 5:
        return "Cemas" if a_sam > 5 else "Tertekan"
    return None


@pytest.mark.parametrize("nama,titik", PUSAT.items())
def test_pusat_kategori(nama, titik):
    assert va_to_mood_prediction(*titik) == nama
    assert decision_margin(*titik, face_conf=0.8) == pytest.approx(0.8)


@pytest.mark.parametrize("titik", BATAS)
def test_tepat_di_batas_bernilai_setengah(titik):
    assert boundary_distance(*titik) == pytest.approx(0.0, abs=1e-12)
    assert decision_margin(*titik) == pytest.approx(0.5)


def test_chebyshev_dan_margin_scale():
    assert boundary_distance(0.1, 0.05) == pytest.approx(0.15)  # di pita: 0,25 - 0,1
    assert boundary_distance(0.30, 0.10) == pytest.approx(0.05)  # ke pita (Chebyshev)
    assert boundary_distance(0.60, 0.05) == pytest.approx(0.05)  # ke garis a = 0
    assert boundary_distance(0.10, 0.60) == pytest.approx(0.10)  # ke garis v = 0
    assert boundary_distance(0.27, 0.27) == pytest.approx(0.02)
    d = boundary_distance(0.6, 0.1)  # 0,1
    assert decision_margin(0.6, 0.1, margin_scale=0.2) == pytest.approx(
        0.5 + 0.5 * d / 0.2
    )
    with pytest.raises(ValueError):
        decision_margin(0.6, 0.1, margin_scale=0.0)


def test_margin_kontinu():
    g = np.linspace(-1, 1, 401)
    s = np.array([[decision_margin(v, a) for v in g] for a in g])
    h = g[1] - g[0]
    assert np.abs(np.diff(s, axis=0)).max() <= 2 * h + 1e-9
    assert np.abs(np.diff(s, axis=1)).max() <= 2 * h + 1e-9


def test_81_pasangan_sam_cocok_tabel33():
    fn = mood_id_function("tabel33", kind="label")
    celah = []
    for v_sam, a_sam in itertools.product(range(1, 10), repeat=2):
        acuan = _tabel33_acuan(v_sam, a_sam)
        got = fn(sam_to_unit(v_sam), sam_to_unit(a_sam))
        if acuan is None:
            assert got == MOOD_GAP_ID
            celah.append((v_sam, a_sam))
        else:
            assert got == MOOD_TO_ID[acuan], (v_sam, a_sam)
    assert celah == [(5, 1), (5, 2), (5, 3), (5, 7), (5, 8), (5, 9)]


def test_prediksi_v_nol_di_luar_pita_sisi_positif():
    fn = mood_id_function("tabel33", kind="prediction")
    assert fn(0.0, 0.75) == MOOD_TO_ID["Ceria"]
    assert fn(0.0, -0.75) == MOOD_TO_ID["Rileks"]
    assert fn(-1e-9, 0.75) == MOOD_TO_ID["Cemas"]


def test_mode_evaluasi_tidak_bergantung_urutan():
    rng = np.random.default_rng(42)
    gt = rng.uniform(-1, 1, (200, 2))
    gt[:10, 0] = 0.0  # sebagian label di celah
    pred = gt + rng.normal(0, 0.15, gt.shape)
    lab = mood_id_function("tabel33", kind="label")
    prd = mood_id_function("tabel33", kind="prediction")
    a = evaluate_va(pred, gt, mood_fn=lab, pred_mood_fn=prd)
    idx = rng.permutation(len(gt))
    b = evaluate_va(pred[idx], gt[idx], mood_fn=lab, pred_mood_fn=prd)
    assert a == pytest.approx(b)
    assert a["mood_n_excluded"] == int(sum(abs(g[1]) > 0.25 for g in gt[:10]))
    ids_maju = [prd(*p) for p in pred]
    ids_mundur = [prd(*p) for p in pred[::-1]][::-1]
    assert ids_maju == ids_mundur


def test_histeresis_tampilan():
    h = MoodDisplayHysteresis(width=0.05)
    assert h.update(0.6, 0.30) == "Ceria"
    assert h.update(0.6, -0.03) == "Ceria"  # masih kandidat: tetap
    assert h.update(0.6, -0.10) == "Rileks"  # di luar lebar histeresis: berganti
    assert h.update(0.0, 0.70) == "Ceria"  # celah: kategori prediksi sisi positif
    h.current = "Cemas"
    assert (
        h.update(0.0, 0.70) == "Cemas"
    )  # celah dengan kategori tampil sebagai kandidat
    tanpa = MoodDisplayHysteresis(width=0.0)
    assert tanpa.update(0.6, 0.02) == "Ceria"
    assert tanpa.update(0.6, -0.02) == "Rileks"


def test_late_fusion_mood_tidak_dipengaruhi_hr_dan_rmssd():
    rng = random.Random(42)
    titik = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(50)]
    hasil = []
    for hr, rmssd in [(55.0, 80.0), (120.0, 12.0), (None, None), (90.0, None)]:
        br = LateFusionVABridge(ema_alpha=0.0)
        out = [
            br.fuse(v, a, timestamp=float(k), hr_bpm=hr, hrv_ms=rmssd)
            for k, (v, a) in enumerate(titik)
        ]
        hasil.append(
            [
                (o["mood_id"], o["decision_margin"], o["valence"], o["arousal"])
                for o in out
            ]
        )
    assert all(h == hasil[0] for h in hasil[1:])


def test_late_fusion_keluaran():
    br = LateFusionVABridge(ema_alpha=0.0)
    out = br.fuse(
        0.6, 0.05, face_confidence=1.0, timestamp=0.0, hr_bpm=80.0, hrv_ms=40.0
    )
    assert "confidence" not in out
    assert out["mood_label"] == "Ceria" and out["decision_margin"] == pytest.approx(0.6)
    assert (
        out["heart_rate_bpm"] == 80.0 and out["hrv_ms"] == 40.0 and out["physio_used"]
    )
    basi = br.fuse(0.6, 0.05, timestamp=10.0)  # data fisiologis kedaluwarsa
    assert basi["heart_rate_bpm"] is None and not basi["physio_used"]
    assert (
        LateFusionVABridge(mood_rule="lingkaran", ema_alpha=0.0).fuse(0.22, 0.22)[
            "mood_label"
        ]
        == "Ceria"
    )
    with pytest.raises(ValueError):
        LateFusionVABridge(mood_rule="kotak")
