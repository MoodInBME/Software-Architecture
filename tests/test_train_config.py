"""Uji pengaturan pelatihan STC-rPPG (ISU-024).

Jalankan dari root repo dengan Python 3.12:

    python -m unittest tests.test_train_config -v
"""

import unittest
from pathlib import Path

import torch

from models.stc_rppg import train as train_mod

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "models" / "stc_rppg" / "config.yaml"


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.cfg = train_mod.load_config(CONFIG)
        torch.manual_seed(42)
        self.model = torch.nn.Linear(4, 2)

    def test_nilai_sesuai_paper(self):
        tr = self.cfg["training"]
        self.assertEqual(tr["optimizer"], "adadelta")
        self.assertEqual(tr["lr"], 1.0)
        self.assertEqual(tr["epochs"], 30)

    def test_optimizer_yang_dipakai(self):
        opt = train_mod.build_optimizer(self.model, self.cfg)
        self.assertIsInstance(opt, torch.optim.Adadelta)
        self.assertEqual(opt.param_groups[0]["lr"], 1.0)

    def test_resume_lr_sama_diterima(self):
        opt = train_mod.build_optimizer(self.model, self.cfg)
        train_mod.check_resume_lr(opt, self.cfg, "uji.pt")

    def test_resume_lr_lama_ditolak(self):
        lama = dict(self.cfg, training=dict(self.cfg["training"], lr=0.0001))
        opt = train_mod.build_optimizer(self.model, lama)
        state = opt.state_dict()
        opt_baru = train_mod.build_optimizer(self.model, self.cfg)
        opt_baru.load_state_dict(state)  # lr lama ikut terpulihkan
        self.assertEqual(opt_baru.param_groups[0]["lr"], 0.0001)
        with self.assertRaises(ValueError):
            train_mod.check_resume_lr(opt_baru, self.cfg, "lama.pt")


if __name__ == "__main__":
    unittest.main()
