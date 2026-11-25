"""Susunan ST-GCN rancangan Pra-TA (B.3, ISU-100): empat unit 64-64-128-128 tanpa Transformer."""

from pathlib import Path

import pytest
import torch

from models.st_gcn_graph_tran import train as train_mod
from models.st_gcn_graph_tran.model import STGCN_GraphTransformer

CONFIG = (
    Path(__file__).resolve().parents[1] / "models" / "st_gcn_graph_tran" / "config.yaml"
)


def test_config_memakai_rancangan_b3():
    cfg = train_mod.load_config(str(CONFIG))
    m = cfg["model"]
    assert m["channels"] == [64, 64, 128, 128] and m["strides"] == [1, 1, 2, 1]
    assert m["use_transformer"] is False
    model = train_mod.build_model(cfg)
    assert model.temporal_transformer is None
    assert [b.stride for b in model.blocks] == [1, 1, 2, 1]
    assert "4 unit" in train_mod.uraian_arsitektur(cfg)


def test_b3_jalan_dengan_mask_dan_lebih_kecil_dari_susunan_lama():
    torch.manual_seed(0)
    cfg = train_mod.load_config(str(CONFIG))
    b3 = train_mod.build_model(cfg).eval()
    lama = STGCN_GraphTransformer().eval()
    n_b3 = sum(p.numel() for p in b3.parameters())
    n_lama = sum(p.numel() for p in lama.parameters())
    assert n_b3 < n_lama / 3
    x = torch.randn(2, 2, 90, 68)
    mask = torch.ones(2, 90)
    mask[1, 60:] = 0
    with torch.no_grad():
        out = b3(x, frame_mask=mask)["va"]
    assert out.shape == (2, 2) and torch.isfinite(out).all()


def test_susunan_lama_tetap_bisa_memuat_bobot_lama():
    lama = STGCN_GraphTransformer(head_type="va")
    baru = STGCN_GraphTransformer(head_type="va")
    baru.load_state_dict(lama.state_dict())
    assert lama.temporal_transformer is not None and len(lama.blocks) == 6


def test_channels_dan_strides_harus_sepadan():
    with pytest.raises(ValueError):
        STGCN_GraphTransformer(channels=(64, 128), strides=(1,))
