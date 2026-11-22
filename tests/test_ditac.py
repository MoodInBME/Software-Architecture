"""DiTAC sesuai Persamaan (1) Lin dkk. 2025: nilai di titik uji, batas domain, dan gradien."""

import math

import pytest
import torch

from models.stc_rppg.ditac import DiTAC


def _gelu(x: float) -> float:
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@pytest.mark.parametrize("x", [-4.0, -2.0, -1.0, 1.0])
def test_inisialisasi_sama_dengan_gelu(x):
    d = DiTAC(num_segments=10, a=-3.0, b=3.0).double()
    y = d(torch.tensor([x], dtype=torch.float64))
    assert y.item() == pytest.approx(_gelu(x), abs=1e-9)


def test_nilai_acuan_tercatat():
    # Nilai acuan ISU-074: -4 -> -0,0, -2 -> -0,046, -1 -> -0,159, 1 -> 0,841.
    d = DiTAC()
    y = d(torch.tensor([-4.0, -2.0, -1.0, 1.0], dtype=torch.float64)).tolist()
    assert [round(v, 3) for v in y] == [-0.0, -0.046, -0.159, 0.841]


def test_sama_dengan_gelu_pytorch_pada_inisialisasi():
    d = DiTAC().double()
    x = torch.linspace(-6, 6, 241, dtype=torch.float64)
    torch.testing.assert_close(d(x), torch.nn.functional.gelu(x), atol=1e-12, rtol=0)
    # float32 (parameter bawaan): selisih hanya pembulatan simpul.
    x32 = x.float()
    torch.testing.assert_close(DiTAC()(x32), torch.nn.functional.gelu(x32), atol=1e-6, rtol=0)


def _acak(seed=0):
    torch.manual_seed(seed)
    d = DiTAC(num_segments=10, a=-3.0, b=3.0)
    with torch.no_grad():
        d.theta_raw.normal_(0.0, 1.0)
    return d


def test_di_luar_domain_x_tilde_sama_dengan_x():
    d = _acak()
    x = torch.tensor([-5.0, -3.5, 3.5, 5.0], dtype=torch.float64)
    phi = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    torch.testing.assert_close(d(x), x * phi)


def test_kontinu_di_batas_domain_dan_monoton():
    d = _acak(1)
    for t in (-3.0, 3.0):
        x = torch.tensor([t - 1e-7, t, t + 1e-7], dtype=torch.float64)
        y = d(x)
        assert (y.max() - y.min()).item() < 1e-5
    u = torch.linspace(0, 1, 1001, dtype=torch.float64)
    tu = d.transform_unit(u)
    assert tu[0].item() == pytest.approx(0.0) and tu[-1].item() == pytest.approx(1.0)
    assert bool((tu[1:] - tu[:-1] >= -1e-12).all())


def test_gradien_ke_parameter_dan_masukan():
    d = DiTAC()
    x = torch.tensor([-2.5, -1.0, 0.3, 2.0, 4.0], requires_grad=True)
    d(x).sum().backward()
    assert d.theta_raw.grad is not None and d.theta_raw.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_domain_tidak_valid_ditolak():
    with pytest.raises(ValueError):
        DiTAC(a=1.0, b=1.0)
