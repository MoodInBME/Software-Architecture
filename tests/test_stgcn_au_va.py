"""Uji head_type au_va dan au_list pada cabang ST-GCN (ISU-071, tugas sesi 8 A.3).

Jalankan dari root repo dengan Python 3.12:

    python -m pytest tests/test_stgcn_au_va.py -q
"""

import copy
from pathlib import Path

import pytest
import torch

from models.st_gcn_graph_tran import train as train_mod
from models.st_gcn_graph_tran.au_va_heads import AUVAHead
from models.st_gcn_graph_tran.heads import VAHead, build_head
from models.st_gcn_graph_tran.model import STGCN_GraphTransformer

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "models" / "st_gcn_graph_tran" / "config.yaml"
DISFA_AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]


def test_config_au_list_dan_bawaan_va():
    cfg = train_mod.load_config(str(CONFIG))
    assert cfg["model"]["au_list"] == DISFA_AUS
    assert cfg["model"]["head_type"] == "va"


def test_build_head_au_va_dan_va_tetap_ada():
    assert isinstance(build_head("va", 256), VAHead)
    head = build_head("au_va", 256, n_au=12)
    assert isinstance(head, AUVAHead)
    out = head(torch.randn(3, 256))
    assert out["au"].shape == (3, 12) and out["va"].shape == (3, 2)
    assert float(out["au"].min()) >= 0.0 and float(out["au"].max()) <= 1.0
    with pytest.raises(ValueError):
        build_head("tidak_ada", 256)


@pytest.mark.parametrize(
    "head_type,keys",
    [
        ("va", {"va"}),
        ("dual", {"va", "logits"}),
        ("au_va", {"au", "va"}),
    ],
)
def test_model_forward_per_head(head_type, keys):
    torch.manual_seed(42)
    model = STGCN_GraphTransformer(head_type=head_type, n_au=len(DISFA_AUS)).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 2, 90, 68))
    assert set(out) == keys
    if "au" in out:
        assert out["au"].shape == (2, len(DISFA_AUS))


def test_build_model_memakai_panjang_au_list():
    cfg = train_mod.load_config(str(CONFIG))
    cfg = copy.deepcopy(cfg)
    cfg["model"]["head_type"] = "au_va"
    cfg["model"]["au_list"] = [4, 6, 12]
    model = train_mod.build_model(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 2, 90, 68))
    assert out["au"].shape == (1, 3)


def test_sanity_au_va(capsys):
    torch.manual_seed(42)
    cfg = copy.deepcopy(train_mod.load_config(str(CONFIG)))
    cfg["model"]["head_type"] = "au_va"
    train_mod.sanity_check(cfg)
    assert "head=au_va" in capsys.readouterr().out
