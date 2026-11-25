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
        "stgcn_va_T48_b3_lipatan"
    )
    assert (tmp_path / "out" / "v2_T48_b3_ringkasan.json").exists()
    assert lap["arsitektur"].startswith("ST-GCN 4 unit") and "tanpa Transformer" in lap["arsitektur"]
    r = lap["per_lipatan"][0]
    assert set(r["sebaran_label"]) == {"Ceria", "Cemas", "Tertekan", "Rileks", "Netral"}
    assert r["mayoritas"]["kelas"] in r["sebaran_label"]


def test_metrik_kelas_hanya_kelas_yang_hadir():
    from scripts.validasi.v2_afew_stgcn import metrik_kelas

    lab = np.array([0, 0, 0, 1, 1, 4])
    pre = np.array([0, 0, 3, 1, 4, 4])
    m = metrik_kelas(lab, pre)
    assert m["kelas_dinilai"] == [0, 1, 4]
    # recall: kelas 0 = 2/3, kelas 1 = 1/2, kelas 4 = 1/1
    assert np.isclose(m["akurasi_seimbang"], (2 / 3 + 1 / 2 + 1) / 3)
    assert 0.0 < m["f1_makro"] < 1.0
    dasar = metrik_kelas(lab, np.zeros_like(lab))
    assert np.isclose(dasar["akurasi_seimbang"], 1 / 3)
    assert metrik_kelas(np.array([], int), np.array([], int))["f1_makro"] is None
