"""
Kerangka pelatihan ST-GCN + Graph-Transformer untuk MoodIn.

CATATAN:
- Ini starter (kerangka awal). Loader dataset nyata dihubungkan melalui
  data/loaders/dataset_loader.py (lihat docs/ST-GCN-PLAN.md).
- Uji bentuk cepat tanpa dataset (butuh PyTorch):
      python train.py --sanity
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

try:
    import yaml
except ImportError:
    yaml = None

try:
    from .model import STGCN_GraphTransformer
    from .labels import MOOD_CLASSES
except ImportError:
    from model import STGCN_GraphTransformer
    from labels import MOOD_CLASSES


def load_config(path):
    default = {
        "model": {"in_channels": 2, "num_class": 5, "num_node": 68,
                  "graph_strategy": "spatial", "max_hop": 1, "t_kernel": 9,
                  "transformer_layers": 2, "transformer_heads": 8, "dropout": 0.3},
        "data": {"window_frames": 150, "fps": 30},
        "train": {"epochs": 80, "batch_size": 16, "lr": 1e-3,
                  "weight_decay": 1e-4, "device": "cuda",
                  "checkpoint_dir": "checkpoints"},
    }
    if path and os.path.exists(path) and yaml is not None:
        with open(path, "r") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            default.setdefault(k, {}).update(v)
    return default


def build_model(cfg):
    m = cfg["model"]
    return STGCN_GraphTransformer(
        in_channels=m["in_channels"], num_class=m["num_class"],
        num_node=m["num_node"], graph_strategy=m["graph_strategy"],
        max_hop=m["max_hop"], t_kernel=m["t_kernel"],
        transformer_layers=m["transformer_layers"],
        transformer_heads=m["transformer_heads"], dropout=m["dropout"],
    )


def sanity_check(cfg):
    """Uji forward + backward dengan data acak (tanpa dataset)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    m, d = cfg["model"], cfg["data"]
    x = torch.randn(4, m["in_channels"], d["window_frames"], m["num_node"]).to(device)
    y = torch.randint(0, m["num_class"], (4,)).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    out = model(x)
    loss = crit(out, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"[sanity] device={device} out={tuple(out.shape)} loss={loss.item():.4f}")
    print(f"[sanity] kelas: {MOOD_CLASSES}")
    print("[sanity] OK: forward + backward berjalan.")


def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    total, correct, running = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = crit(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return running / max(total, 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sanity", action="store_true",
                    help="Jalankan uji bentuk cepat tanpa dataset")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.sanity:
        sanity_check(cfg)
        return

    # === Pelatihan penuh ===
    # TODO: hubungkan dataset nyata di sini.
    #   from data.loaders.dataset_loader import MoodLandmarkDataset
    #   train_ds = MoodLandmarkDataset(split="train", cfg=cfg)
    #   train_loader = torch.utils.data.DataLoader(train_ds, ...)
    raise NotImplementedError(
        "Loader dataset belum dihubungkan. Jalankan `--sanity` untuk uji model, "
        "atau implementasikan data/loaders/dataset_loader.py lebih dulu "
        "(lihat docs/ST-GCN-PLAN.md)."
    )


if __name__ == "__main__":
    main()
