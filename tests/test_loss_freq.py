"""Uji rugi domain frekuensi STC-rPPG (tugas 1b).

Pilihan bawaan freq_loss="mae_hr" mengikuti Lin dkk. (2025, Persamaan 9) dan
rancangan Pra-TA Persamaan 3.10-3.12: L_freq = |HR_pre - HR_gt| dalam bpm,
dengan HR_pre dari soft-argmax spektral. Pilihan "kl_psd" adalah rugi versi
GEMASTIK yang dipertahankan untuk perbandingan.

Jalankan dari root repo dengan Python 3.12:

    python -m unittest tests.test_loss_freq -v
"""

import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from models.stc_rppg import train as train_mod
from models.stc_rppg.loss import FREQ_LOSSES, STCHybridLoss

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0
T = 128  # panjang klip STC-rPPG
SEED = 42


def _sinus(bpm, n=T, noise=0.0, gen=None):
    t = torch.arange(n, dtype=torch.float32) / FPS
    x = torch.sin(2 * math.pi * (bpm / 60.0) * t)
    if noise:
        x = x + noise * torch.randn(n, generator=gen)
    return x.unsqueeze(0)


class TestPilihan(unittest.TestCase):
    def test_bawaan_mae_hr(self):
        crit = STCHybridLoss()
        self.assertEqual(crit.freq_loss, "mae_hr")
        self.assertEqual(FREQ_LOSSES, ("mae_hr", "kl_psd"))

    def test_pilihan_tidak_valid_ditolak(self):
        with self.assertRaises(ValueError):
            STCHybridLoss(freq_loss="mse_hr")
        for tau in (0.0, -0.1):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                STCHybridLoss(tau=tau)


class TestEstimasiHR(unittest.TestCase):
    def setUp(self):
        self.crit = STCHybridLoss(fps=FPS)
        self.bpms = [50.0 + 7.3 * i for i in range(25)]  # 50-225 bpm, tidak tepat di bin

    def test_soft_hr_sinus_murni(self):
        for bpm in self.bpms:
            with self.subTest(bpm=bpm):
                self.assertAlmostEqual(float(self.crit.soft_hr(_sinus(bpm))), bpm, delta=0.5)

    def test_soft_hr_berderau(self):
        gen = torch.Generator().manual_seed(SEED)
        galat = [abs(float(self.crit.soft_hr(_sinus(b, noise=0.5, gen=gen))) - b) for b in self.bpms]
        self.assertLess(sum(galat) / len(galat), 1.5)
        self.assertLess(max(galat), 3.0)

    def test_bentuk_per_sampel(self):
        x = torch.cat([_sinus(60.0), _sinus(90.0), _sinus(150.0)])
        hr = self.crit.soft_hr(x)
        self.assertEqual(tuple(hr.shape), (3,))
        for got, want in zip(hr.tolist(), (60.0, 90.0, 150.0)):
            self.assertAlmostEqual(got, want, delta=0.5)


class TestRugiMaeHR(unittest.TestCase):
    def setUp(self):
        self.crit = STCHybridLoss(fps=FPS)

    def test_selisih_dalam_bpm(self):
        _, _, l_freq = self.crit(_sinus(60.0), _sinus(90.0))
        self.assertAlmostEqual(float(l_freq), 30.0, delta=0.5)

    def test_sinyal_identik(self):
        x = _sinus(84.0, noise=0.1, gen=torch.Generator().manual_seed(SEED))
        total, l_time, l_freq = self.crit(x, x.clone())
        self.assertLess(float(l_time), 1e-5)
        self.assertLess(float(l_freq), 1e-4)
        self.assertAlmostEqual(float(total), float(l_time) + 0.5 * float(l_freq), places=5)

    def test_gradien_mengarah_ke_hr_acuan(self):
        torch.manual_seed(SEED)
        acuan = _sinus(100.0)
        pred = (_sinus(80.0) + 0.05 * torch.randn(1, T)).requires_grad_(True)
        opt = torch.optim.Adam([pred], lr=0.01)
        awal = float(self.crit(pred, acuan)[2])
        for _ in range(200):
            opt.zero_grad()
            _, _, l_freq = self.crit(pred, acuan)
            l_freq.backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
            opt.step()
        akhir = float(self.crit(pred, acuan)[2])
        self.assertGreater(awal, 15.0)
        self.assertLess(akhir, 0.5 * awal)

    def test_acuan_tanpa_gradien(self):
        acuan = _sinus(100.0).requires_grad_(True)
        pred = _sinus(80.0).requires_grad_(True)
        _, _, l_freq = self.crit(pred, acuan)
        l_freq.backward()
        self.assertIsNone(acuan.grad)
        self.assertIsNotNone(pred.grad)

    def test_masukan_half_precision(self):
        _, _, l_freq = self.crit(_sinus(60.0).half(), _sinus(90.0).half())
        self.assertEqual(l_freq.dtype, torch.float32)
        self.assertAlmostEqual(float(l_freq), 30.0, delta=1.0)


class TestRugiKlPsd(unittest.TestCase):
    def setUp(self):
        self.crit = STCHybridLoss(fps=FPS, freq_loss="kl_psd")

    def test_sinyal_identik(self):
        x = _sinus(84.0)
        _, _, l_freq = self.crit(x, x.clone())
        self.assertLess(float(l_freq), 1e-5)

    def test_sama_dengan_kl_div_psd(self):
        p, q = _sinus(60.0), _sinus(90.0)
        _, _, l_freq = self.crit(p, q)
        harap = F.kl_div(self.crit._band_logpsd(p), self.crit._band_logpsd(q).exp(), reduction="batchmean")
        self.assertAlmostEqual(float(l_freq), float(harap), places=6)
        self.assertGreater(float(l_freq), 0.0)


class TestConfig(unittest.TestCase):
    def test_config_repo(self):
        cfg = train_mod.load_config(REPO / "models" / "stc_rppg" / "config.yaml")
        crit = train_mod.build_criterion(cfg)
        self.assertEqual(crit.freq_loss, "mae_hr")
        self.assertEqual(crit.tau, cfg["loss"]["softargmax_tau"])
        self.assertEqual(crit.n_fft, cfg["loss"]["n_fft"])

    def test_pilihan_kl_psd_dari_config(self):
        crit = train_mod.build_criterion({"loss": {"fps": 30.0, "freq_loss": "kl_psd"}})
        self.assertEqual(crit.freq_loss, "kl_psd")

    def test_config_tanpa_kunci_memakai_bawaan(self):
        crit = train_mod.build_criterion({"loss": {"fps": 30.0}, "training": {}})
        bawaan = STCHybridLoss()
        self.assertEqual((crit.freq_loss, crit.tau, crit.n_fft, crit.lambda_weight),
                         (bawaan.freq_loss, bawaan.tau, bawaan.n_fft, bawaan.lambda_weight))

    def test_nilai_salah_di_config_ditolak(self):
        with self.assertRaises(ValueError):
            train_mod.build_criterion({"loss": {"freq_loss": "kl"}})


if __name__ == "__main__":
    unittest.main()
