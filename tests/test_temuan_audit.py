"""Uji temuan audit 22 Nov: peta landmark, skala AU, CSV OpenFace gambar, padding dan mask."""

import numpy as np
import pytest
import torch

from data.loaders.affect_landmark_dataset import AffectLandmarkDataset
from data.loaders.openface_csv import (
    best_face,
    read_openface_csv,
    read_openface_image_csv,
)
from data.preprocessing import landmark_extraction_mediapipe as lem
from data.preprocessing import landmark_map as lmap
from data.preprocessing.au_scale import ke_satuan, ke_skala_asli, periksa_satuan
from evaluation.affect_metrics import au_report
from models.st_gcn_graph_tran.model import STGCN_GraphTransformer
from scripts.validasi.laporan_padding_afew import ringkas_padding


# ------------------------------------------------------------- 3.1 peta landmark
def test_dua_varian_satu_sumber_beda_di_0_17_26():
    a, b = lmap.peta_ibug68("A"), lmap.peta_ibug68("B")
    assert (
        [i for i in range(68) if a[i] != b[i]]
        == [0, 17, 26]
        == list(lmap.TITIK_BERBEDA)
    )
    assert (a[0], a[17], a[26]) == (162, 71, 301) and (b[0], b[17], b[26]) == (
        127,
        70,
        300,
    )
    # Pemanggil lama tidak berubah perilaku.
    assert lmap.MEDIAPIPE_FACEMESH_TO_IBUG68 == a
    assert (
        lem.MEDIAPIPE_TO_IBUG68 == b
        and lem.MEDIAPIPE_TO_IBUG68 is lmap.MEDIAPIPE_TO_IBUG68
    )


def test_subset_per_varian():
    pts = np.arange(468 * 2, dtype=np.float32).reshape(468, 2)
    sa, sb = lmap.subset_to_ibug68(pts, "A"), lmap.subset_to_ibug68(pts, "B")
    assert sa.shape == sb.shape == (68, 2)
    np.testing.assert_array_equal(sb, lem.mediapipe_to_ibug68(pts))
    with pytest.raises(ValueError):
        lmap.peta_ibug68("C")


# ------------------------------------------------------------- 3.2 skala AU
def test_konversi_skala_au_dua_arah():
    x = np.array([0.0, 2.5, 5.0, np.nan])
    u = ke_satuan(x)
    np.testing.assert_allclose(u[:3], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(ke_skala_asli(u)[:3], x[:3])
    with pytest.raises(ValueError):
        ke_satuan([6.0])
    with pytest.raises(ValueError):
        periksa_satuan([0.2, 3.0])


def test_au_report_menangkap_salah_skala():
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 1, (50, 2))
    p = np.clip(t + 0.1, 0, 1)
    r = au_report(t, p, [1, 12])
    assert r["skala"] == "0..5"
    assert r["AU1"]["mae"] == pytest.approx(np.abs((p - t)[:, 0]).mean() * 5)
    # Keluaran OpenFace AUxx_r mentah (0..5) yang lupa dikonversi harus ditolak.
    with pytest.raises(ValueError, match="0..5"):
        au_report(t * 5, p, [1, 12])


# ------------------------------------------------------------- 3.3 OpenFace gambar
def _csv_gambar(path, baris, dengan_conf=True):
    cols = ["face"] + (["confidence"] if dengan_conf else [])
    cols += [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)] + ["AU12_r"]
    teks = ", ".join(cols) + "\n"
    for face, conf, val in baris:
        r = [face] + ([conf] if dengan_conf else []) + [val] * 136 + [2.5]
        teks += ", ".join(map(str, r)) + "\n"
    path.write_text(teks)
    return str(path)


def test_csv_facelandmarkimg_tanpa_frame_dan_success(tmp_path):
    p = _csv_gambar(tmp_path / "a.csv", [(0, 0.7, 10.0), (1, 0.95, 20.0)])
    r = read_openface_image_csv(p)
    assert r["n_wajah"] == 2 and r["landmarks"].shape == (2, 68, 2)
    assert best_face(r)[0, 0] == 20.0
    assert best_face(r, min_confidence=0.99) is None
    assert ke_satuan(r["au"][12])[0] == pytest.approx(0.5)
    with pytest.raises(KeyError, match="read_openface_image_csv"):
        read_openface_csv(p)


def test_csv_gambar_tanpa_wajah_dan_tanpa_confidence(tmp_path):
    kosong = _csv_gambar(tmp_path / "k.csv", [])
    assert read_openface_image_csv(kosong)["n_wajah"] == 0
    assert best_face(read_openface_image_csv(kosong)) is None
    tanpa = _csv_gambar(tmp_path / "t.csv", [(0, None, 5.0)], dengan_conf=False)
    assert best_face(read_openface_image_csv(tanpa))[0, 0] == 5.0


# ------------------------------------------------------------- 3.4 padding dan mask
def test_ringkasan_padding():
    r = ringkas_padding([45, 90, 120], clip_len=90, stride=15)
    # 45 -> 1 jendela (45 frame pad); 90 -> 1; 120 -> 0, 15, 30 = 3 jendela.
    assert (r["jendela"], r["jendela_berpadding"], r["frame_padding"]) == (5, 1, 45)
    assert r["persen_frame_padding"] == pytest.approx(10.0)
    assert r["klip_pendek"] == 1


def test_dataset_memberi_mask_frame():
    x = torch.arange(2 * 40 * 68, dtype=torch.float32).reshape(2, 40, 68)
    y, m = AffectLandmarkDataset._fix_len_mask(x, 90)
    assert y.shape == (2, 90, 68) and m.sum().item() == 40 and m[39] == 1 and m[40] == 0
    torch.testing.assert_close(y[:, 89], x[:, 39])  # pad = frame terakhir
    _, m2 = AffectLandmarkDataset._fix_len_mask(torch.zeros(2, 120, 68), 90)
    assert m2.sum().item() == 90


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return STGCN_GraphTransformer(head_type="va").eval()


@pytest.mark.parametrize("T", [10, 45, 89, 90, 145])
def test_panjang_mask_turun_sama_dengan_fitur(model, T):
    x = torch.randn(1, 2, T, 68)
    feats = x.permute(0, 3, 1, 2).reshape(1, 136, T)
    feats = model.data_bn(feats).view(1, 68, 2, T).permute(0, 2, 3, 1)
    with torch.no_grad():
        for blk in model.blocks:
            feats = blk(feats, model.A)
    assert model.downsample_mask(torch.ones(1, T)).shape[1] == feats.shape[2]


def test_pooling_mengabaikan_posisi_padding(model):
    torch.manual_seed(1)
    f = torch.randn(2, 23, 256)
    mask = torch.zeros(2, 23, dtype=torch.bool)
    mask[0, :10] = True
    mask[1, :] = True
    g = f.clone()
    g[0, 10:] = 100.0  # ubah hanya posisi padding sampel 0
    with torch.no_grad():
        a, b = model.pool_temporal(f, mask), model.pool_temporal(g, mask)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
    with torch.no_grad():
        c, d = model.pool_temporal(f, None), model.pool_temporal(g, None)
    assert (c[0] - d[0]).abs().max() > 1e-3  # tanpa mask, padding ikut memengaruhi


def test_mask_penuh_sama_dengan_tanpa_mask(model):
    x = torch.randn(2, 2, 90, 68)
    with torch.no_grad():
        a = model(x)["va"]
        b = model(x, frame_mask=torch.ones(2, 90))["va"]
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
