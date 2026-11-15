"""Uji pengaturan pelatihan STC-rPPG (ISU-024).

Jalankan dari root repo dengan Python 3.12:

    python -m unittest tests.test_train_config -v
"""

import csv
import tempfile
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


class TestRiwayatPelatihan(unittest.TestCase):
    def test_header_sekali_dan_kolom_terpisah(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / train_mod.HISTORY_FILE
            for ep in (1, 2):
                train_mod.append_history_csv(path, {
                    "epoch": ep, "lr": 1.0, "freq_loss": "mae_hr",
                    "train_loss": 10.6, "train_loss_time": 1.0, "train_loss_freq": 19.2,
                    "train_data_time": 3.0,  # kolom tambahan diabaikan
                    "val_loss": 9.0, "val_loss_time": 0.9, "val_loss_freq": 16.2,
                    "val_hr_mae": 15.0, "val_hr_rmse": 20.0,
                })
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(tuple(rows[0].keys()), train_mod.HISTORY_FIELDS)
            self.assertEqual(rows[1]["epoch"], "2")
            self.assertEqual(float(rows[0]["train_loss_freq"]), 19.2)
            self.assertEqual(float(rows[0]["train_loss_time"]), 1.0)


if __name__ == "__main__":
    unittest.main()
