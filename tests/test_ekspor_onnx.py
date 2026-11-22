"""V5: CDC sama dengan satu konvolusi berbobot gabungan, dan ST-GCN dapat diekspor ke ONNX."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.stc_rppg.cdc import CDC_ST


def test_cdc_setara_satu_konvolusi_biasa():
    # CDC = conv(x, W) - theta * conv1x1(x, sum W) = conv(x, W') dengan W' = W - theta * sum(W) di pusat kernel.
    torch.manual_seed(0)
    m = CDC_ST(4, 6, theta=0.6).eval()
    x = torch.randn(2, 4, 8, 12, 12)
    w = m.weight.detach().clone()
    w[:, :, 1, 1, 1] -= 0.6 * m.weight.detach().sum(dim=(2, 3, 4))
    y = F.conv3d(x, w, m.bias, m.stride, m.padding)
    torch.testing.assert_close(m(x), y, atol=1e-5, rtol=0)


def test_stgcn_ekspor_onnx_sama_dengan_pytorch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from scripts.validasi.v5_ekspor_onnx import _KeluaranVA, ekspor, model_stgcn

    torch.manual_seed(0)
    m, x = model_stgcn(None)
    lap = ekspor("stgcn_uji", _KeluaranVA(m), x, tmp_path)
    assert lap["bentuk_keluaran"] == [1, 2] and lap["selisih_maks"] < 1e-4
    assert ort is not None and np.isfinite(lap["ms_ort_cpu_satu_kali"])
