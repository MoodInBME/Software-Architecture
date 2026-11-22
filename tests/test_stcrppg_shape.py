"""Uji bentuk STC-rPPG (pengganti tests/main.py yang memakai impor sebelum reorganisasi)."""

import torch

from models.stc_rppg import STCrPPG


def test_bentuk_keluaran_dan_jumlah_parameter():
    torch.manual_seed(0)
    model = STCrPPG().eval()
    x = torch.randn(1, 3, 128, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 128)
    assert torch.isfinite(y).all()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Dicatat apa adanya; bukan klaim kesetaraan dengan jumlah parameter paper.
    assert n == 245_942
