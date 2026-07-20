"""
Pelatihan cabang ST-GCN (VA-bridge). Menggantikan kerangka awal dengan pipeline
lengkap: dataset landmark ter-cache -> regresi VA (loss CCC) -> eval (CCC/F1).

Jalankan sebagai modul dari root repo:
    python -m models.st_gcn_graph_tran.train \
        --config models/st_gcn_graph_tran/config.yaml \
        --manifest data/prepared/ckplus/manifest.json

Uji cepat tanpa dataset (data acak):
    python -m models.st_gcn_graph_tran.train --sanity
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Pastikan root repo ada di sys.path saat dijalankan sebagai skrip.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import yaml
except ImportError:
    yaml = None

from models.st_gcn_graph_tran.model import STGCN_GraphTransformer
from models.st_gcn_graph_tran.losses import build_loss
from evaluation.va_metrics import evaluate_va


def load_config(path):
    default = {
        "model": {"in_channels": 2, "num_class": 5, "num_node": 68,
                  "graph_strategy": "spatial", "max_hop": 1, "t_kernel": 9,
                  "transformer_layers": 2, "transformer_heads": 8,
                  "transformer_ff": 512, "dropout": 0.3, "head_type": "va"},
        "data": {"window_frames": 150, "fps": 30, "neutral_radius": 0.2},
        "loss": {"alpha_va": 1.0, "beta_cls": 0.5},
        "train": {"epochs": 80, "batch_size": 16, "lr": 1e-3,
                  "weight_decay": 1e-4, "device": "cuda",
                  "checkpoint_dir": "checkpoints", "num_workers": 2},
    }
    if path and os.path.exists(path) and yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
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
        transformer_heads=m["transformer_heads"],
        transformer_ff=m.get("transformer_ff", 512),
        dropout=m["dropout"], head_type=m.get("head_type", "va"),
    )


def _build_loss(cfg, head_type):
    if head_type == "dual":
        return build_loss("dual", alpha=cfg["loss"]["alpha_va"], beta=cfg["loss"]["beta_cls"])
    return build_loss(head_type)


def sanity_check(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    head_type = cfg["model"].get("head_type", "va")
    loss_fn = _build_loss(cfg, head_type)
    m, d = cfg["model"], cfg["data"]
    x = torch.randn(4, m["in_channels"], d["window_frames"], m["num_node"]).to(device)
    tgt_va = (torch.rand(4, 2).to(device) * 2 - 1)
    tgt_cls = torch.randint(0, m["num_class"], (4,)).to(device)
    model.train()
    out = model(x)
    if head_type == "va":
        loss = loss_fn(out["va"], tgt_va)
    elif head_type == "dual":
        loss = loss_fn(out, tgt_va, tgt_cls)
    else:
        loss = loss_fn(out["logits"], tgt_cls)
    loss.backward()
    shapes = {k: tuple(v.shape) for k, v in out.items()}
    print(f"[sanity] device={device} head={head_type} out={shapes} loss={float(loss):.4f} OK")


def run_epoch(model, loader, loss_fn, device, head_type, optimizer=None):
    train = optimizer is not None
    model.train(train)
    preds, tgt_va, tgt_mood, total = [], [], [], 0.0
    for x, target_va, mood in loader:
        x = x.to(device); target_va = target_va.to(device); mood = mood.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            if head_type == "va":
                loss = loss_fn(out["va"], target_va)
            elif head_type == "dual":
                loss = loss_fn(out, target_va, mood)
            else:
                loss = loss_fn(out["logits"], mood)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += float(loss) * x.size(0)
        if "va" in out:
            preds.append(out["va"].detach().cpu().numpy())
            tgt_va.append(target_va.cpu().numpy())
            tgt_mood.append(mood.cpu().numpy())
    metrics = {}
    if preds:
        metrics = evaluate_va(np.concatenate(preds), np.concatenate(tgt_va), np.concatenate(tgt_mood))
    metrics["loss"] = total / len(loader.dataset)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/st_gcn_graph_tran/config.yaml")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--val_split", type=float, default=0.2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.sanity:
        sanity_check(cfg)
        return
    if not args.manifest:
        raise SystemExit("--manifest wajib (atau pakai --sanity).")

    from torch.utils.data import DataLoader, random_split
    from data.loaders.expression_video_loader import ExpressionLandmarkDataset

    tr = cfg["train"]
    device = tr["device"] if torch.cuda.is_available() else "cpu"
    head_type = cfg["model"].get("head_type", "va")

    ds = ExpressionLandmarkDataset(
        args.manifest, clip_len=cfg["data"]["window_frames"],
        neutral_radius=cfg["data"].get("neutral_radius", 0.2),
    )
    n_val = max(1, int(len(ds) * args.val_split))
    tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val],
                               generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr_ds, batch_size=tr["batch_size"], shuffle=True,
                           num_workers=tr.get("num_workers", 2), drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=tr["batch_size"], shuffle=False,
                           num_workers=tr.get("num_workers", 2))

    model = build_model(cfg).to(device)
    loss_fn = _build_loss(cfg, head_type)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr["epochs"])

    ckpt_dir = Path(tr["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(tr["epochs"]):
        tr_m = run_epoch(model, tr_loader, loss_fn, device, head_type, opt)
        va_m = run_epoch(model, va_loader, loss_fn, device, head_type)
        sched.step()
        score = 0.5 * (va_m.get("ccc_valence", 0) + va_m.get("ccc_arousal", 0))
        print(f"epoch {epoch+1}/{tr['epochs']} | train_loss {tr_m['loss']:.4f} "
              f"| val_loss {va_m['loss']:.4f} | val_CCC {score:.4f} "
              f"| val_moodF1 {va_m.get('mood_macro_f1', 0):.4f}")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch},
                       ckpt_dir / "stgcn_va_best.pth")
    print(f"Selesai. Best val CCC={best:.4f} -> {ckpt_dir/'stgcn_va_best.pth'}")


if __name__ == "__main__":
    main()
