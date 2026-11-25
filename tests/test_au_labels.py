"""Peta indeks AU kepala (AU1, AU4, AU6, AU12) ke kolom DISFA dan Aff-Wild2."""

import json

import numpy as np
import pytest
import torch

from data.loaders import disfa
from data.loaders.affect_landmark_dataset import AffectLandmarkDataset
from models.st_gcn_graph_tran import au_labels as L
from models.st_gcn_graph_tran import train as train_mod


def test_au_ta_sesuai_bimbingan():
    assert L.AU_TA == (1, 4, 6, 12)
    assert 7 not in L.AU_TA


def test_daftar_disfa_sama_dengan_loader():
    # Dua salinan daftar kolom DISFA harus identik; bila berbeda, peta indeks salah diam-diam.
    assert list(L.DISFA_AUS) == list(disfa.DISFA_AUS)


def test_peta_indeks_disfa_dan_affwild2():
    assert L.peta_indeks(L.AU_TA, L.DISFA_AUS) == [0, 2, 4, 6]
    assert L.peta_indeks(L.AU_TA, L.AFFWILD2_AUS) == [0, 2, 3, 6]
    assert L.SUMBER["disfa"].jenis == L.INTENSITAS
    assert L.SUMBER["affwild2"].jenis == L.KEMUNCULAN


def test_peta_indeks_mengikuti_nomor_bukan_posisi():
    assert L.peta_indeks([12, 1], [1, 12, 4]) == [1, 0]
    assert L.peta_indeks([7], [1, 4]) == [None]
    with pytest.raises(ValueError):
        L.peta_indeks([1], [1, 1, 4])


def test_validasi_menolak_au_tanpa_label():
    peta = L.validasi_au_list(L.AU_TA)
    assert set(peta) == {"disfa", "affwild2"}
    with pytest.raises(ValueError, match="tidak punya label"):
        L.validasi_au_list([1, 7], sumber=("disfa",))
    with pytest.raises(ValueError):
        L.validasi_au_list([1, 1])
    with pytest.raises(ValueError):
        L.validasi_au_list([1], sumber=("bp4d",))


def test_build_model_au_va_menolak_au_yatim():
    cfg = train_mod.load_config(None)
    cfg["model"]["head_type"] = "au_va"
    cfg["model"]["au_list"] = [1, 7]
    cfg["model"]["au_sumber"] = ["disfa"]
    with pytest.raises(ValueError):
        train_mod.build_model(cfg)
    cfg["model"]["au_list"] = list(L.AU_TA)
    model = train_mod.build_model(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 2, 90, 68))
    assert out["au"].shape == (1, 4)


def test_dataset_memetakan_kolom_disfa_ke_au_ta(tmp_path):
    # Kolom DISFA diisi nomor AU / 100 agar posisi yang tertukar langsung terlihat.
    nilai = [a / 100 for a in disfa.DISFA_AUS]
    T = 30
    torch.save({"landmarks": torch.zeros(2, T, 68)}, tmp_path / "k0.pt")
    man = [
        {
            "file": "k0.pt",
            "subject": "s1",
            "va_mask": 0.0,
            "au": nilai,
            "au_mask": [1.0] * 12,
        }
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    (tmp_path / "meta.json").write_text(json.dumps({"au_list": disfa.DISFA_AUS}))
    ds = AffectLandmarkDataset(
        str(tmp_path / "manifest.json"), au_list=list(L.AU_TA), clip_len=T
    )
    _, _, _, au, am = ds[0]
    assert am.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert np.allclose(au.numpy(), [0.01, 0.04, 0.06, 0.12])
