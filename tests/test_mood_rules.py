"""Uji pilihan aturan mood (tabel33 dan lingkaran) untuk evaluasi dan pelatihan ST-GCN.

Jalankan dari root repo dengan Python 3.12:

    python -m pytest tests/test_mood_rules.py -q
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from evaluation.va_metrics import evaluate_va
from models.st_gcn_graph_tran import train as train_mod
from models.st_gcn_graph_tran.labels import MOOD_TO_ID
from models.st_gcn_graph_tran.losses import GapAwareCrossEntropy, build_loss
from models.st_gcn_graph_tran.mood_rules import (
    MOOD_GAP_ID,
    MoodRuleDataset,
    mood_id_function,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "models" / "st_gcn_graph_tran" / "config.yaml"


def test_tabel33_berbeda_dari_lingkaran():
    t33, ling = mood_id_function("tabel33"), mood_id_function("lingkaran", 0.2)
    assert t33(0.22, 0.22) == MOOD_TO_ID["Netral"]
    assert ling(0.22, 0.22) == MOOD_TO_ID["Ceria"]


def test_celah_aturan_diberi_id_gap():
    assert mood_id_function("tabel33")(0.0, 0.75) == MOOD_GAP_ID
    assert mood_id_function("lingkaran")(0.0, 0.75) == MOOD_TO_ID["Ceria"]


def test_aturan_tidak_dikenal_ditolak():
    with pytest.raises(ValueError):
        mood_id_function("kotak")


def test_config_bawaan_tabel33():
    cfg = train_mod.load_config(str(CONFIG))
    assert cfg["data"]["mood_rule"] == "tabel33"
    assert train_mod.build_mood_fn(cfg)(0.22, 0.22) == MOOD_TO_ID["Netral"]


def test_evaluate_va_mengeluarkan_celah():
    gt = np.array([[0.6, 0.6], [-0.6, 0.6], [0.0, 0.75], [0.1, 0.1]])
    out = evaluate_va(gt.copy(), gt, mood_fn=mood_id_function("tabel33"))
    assert out["mood_n_excluded"] == 1
    assert out["mood_accuracy"] == 1.0


def test_evaluate_va_tanpa_mood_fn_tetap_lingkaran():
    gt = np.array([[0.22, 0.22], [-0.6, -0.6]])
    out = evaluate_va(gt.copy(), gt)
    assert out["mood_n_excluded"] == 0 and out["mood_accuracy"] == 1.0


def test_cross_entropy_mengabaikan_celah():
    torch.manual_seed(42)
    logits = torch.randn(4, 5, requires_grad=True)
    target = torch.tensor([0, MOOD_GAP_ID, 3, MOOD_GAP_ID])
    loss = GapAwareCrossEntropy()(logits, target)
    harap = F.cross_entropy(logits[[0, 2]], target[[0, 2]])
    assert torch.isclose(loss, harap)
    semua_celah = GapAwareCrossEntropy()(logits, torch.full((4,), MOOD_GAP_ID))
    assert semua_celah.item() == 0.0
    (grad,) = torch.autograd.grad(semua_celah, logits)
    assert torch.isfinite(grad).all()
    assert isinstance(build_loss("class"), GapAwareCrossEntropy)


class _Dasar(torch.utils.data.Dataset):
    def __init__(self):
        self.va = [(0.22, 0.22), (0.0, 0.75), (-0.6, -0.6)]

    def __len__(self):
        return len(self.va)

    def __getitem__(self, i):
        v, a = self.va[i]
        return torch.zeros(2, 90, 68), torch.tensor([v, a]), torch.tensor(0)


def test_dataset_dibungkus_memakai_aturan_baru():
    ds = MoodRuleDataset(_Dasar(), mood_id_function("tabel33"))
    moods = [int(ds[i][2]) for i in range(len(ds))]
    assert moods == [MOOD_TO_ID["Netral"], MOOD_GAP_ID, MOOD_TO_ID["Tertekan"]]
    x, va, _ = ds[0]
    assert x.shape == (2, 90, 68) and torch.allclose(va, torch.tensor([0.22, 0.22]))
