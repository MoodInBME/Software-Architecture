"""V2: uji asap pelatihan GroupKFold pada klip AFEW-VA sintetis (tanpa data asli)."""

import argparse
import json

import numpy as np
import pytest
import torch

from data.preprocessing.affect_windows import make_clips, save_clips
from scripts.validasi.v2_afew_stgcn import jalankan, rugi_va


def test_rugi_va_nol_saat_prediksi_tepat():
    t = torch.tensor([[0.1, -0.2], [0.5, 0.3], [-0.4, 0.6]])
    assert rugi_va(t, t).item() == pytest.approx(0.0, abs=1e-6)
    assert rugi_va(-t, t).item() > 1.0


def _siapkan(folder, n_aktor=6, panjang=(30, 60)):
    rng = np.random.default_rng(0)
    manifest = []
    for a in range(n_aktor):
        for c, T in enumerate(panjang):
            lm = rng.normal(0, 1, (T, 68, 2)).astype(np.float32)
            va = np.tile(rng.uniform(-0.9, 0.9, 2).astype(np.float32), (T, 1))
            recs = make_clips(
                lm,
                np.zeros(T, bool),
                f"a{a}_c{c}",
                f"aktor{a}",
                "afew-va",
                48,
                15,
                va=va,
            )
            save_clips(recs, str(folder / "clips"), manifest)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder / "manifest.json"


def test_v2_jalan_ujung_ke_ujung_di_cpu(tmp_path):
    man = _siapkan(tmp_path / "prep")
    args = argparse.Namespace(
        manifest=str(man),
        T=48,
        keluaran=str(tmp_path / "out"),
        lipatan=3,
        epoch=1,
        batch=4,
        lr=1e-3,
        seed=42,
        cpu=True,
    )
    lap = jalankan(args)
    assert len(lap["per_lipatan"]) == 3 and lap["jumlah_aktor"] == 6
    for r in lap["per_lipatan"]:
        assert r["aktor_uji"] == 2 and np.isfinite(r["ccc_valensi"])
        assert r["akurasi_5_kelas"] is None or 0.0 <= r["akurasi_5_kelas"] <= 1.0
    assert sorted(p.name for p in (tmp_path / "out").glob("*.pt"))[0].startswith(
        "stgcn_va_T48_lipatan"
    )
    assert (tmp_path / "out" / "v2_T48_ringkasan.json").exists()
