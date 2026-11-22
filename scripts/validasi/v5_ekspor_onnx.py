"""V5: ekspor ST-GCN dan STC-rPPG ke ONNX (opset 17) dan cek kesamaan keluaran dengan ONNX Runtime.

Bobot terlatih dibaca dari checkpoint di luar repo bila diberikan; tanpa checkpoint,
model diinisialisasi acak (cukup untuk memeriksa bahwa graf dapat diekspor).
Keluaran .onnx dan laporan JSON ditulis ke folder di luar repo.

    python -m scripts.validasi.v5_ekspor_onnx --keluaran D:/moodin_runs/v5
    python -m scripts.validasi.v5_ekspor_onnx --keluaran D:/moodin_runs/v5 --stgcn-ckpt D:/moodin_runs/v2/lipatan0.pt
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from scripts.validasi._umum import REPO, info_lingkungan, pastikan_di_luar_repo, tulis_json

OPSET = 17


def model_stgcn(ckpt: str | None, T: int = 90):
    import yaml

    from models.st_gcn_graph_tran.train import build_model

    with open(REPO / "models/st_gcn_graph_tran/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = build_model(cfg)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        m.load_state_dict(sd.get("model", sd))
    return m.eval(), torch.randn(
        1, cfg["model"]["in_channels"], T, cfg["model"]["num_node"]
    )


def model_stcrppg(ckpt: str | None):
    from models.stc_rppg import STCrPPG

    m = STCrPPG()
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        m.load_state_dict(sd.get("model", sd))
    return m.eval(), torch.randn(1, 3, 128, 128, 128)


class _KeluaranVA(torch.nn.Module):
    """ST-GCN mengembalikan dict; ONNX butuh tensor."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out["va"] if "va" in out else out["logits"]


def ekspor(nama: str, model, contoh: torch.Tensor, folder) -> dict:
    import onnx
    import onnxruntime as ort

    path = folder / f"{nama}.onnx"
    with torch.no_grad():
        acuan = model(contoh).numpy()
    # Jalur cepat TransformerEncoder (aktif di no_grad) tidak bisa diekspor ke ONNX.
    torch.backends.mha.set_fastpath_enabled(False)
    # Tanpa no_grad, pelacakan STC-rPPG 128 x 128 x 128 menyimpan graf autograd dan
    # kehabisan memori (8 GB di cloud); dengan no_grad puncak memori sekitar 2,2 GB.
    with torch.no_grad():
        torch.onnx.export(
            model,
            (contoh,),
            str(path),
            opset_version=OPSET,
            input_names=["masukan"],
            output_names=["keluaran"],
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(str(path)))
    sesi = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    t = time.perf_counter()
    hasil = sesi.run(None, {"masukan": contoh.numpy()})[0]
    ms = 1000 * (time.perf_counter() - t)
    return {
        "berkas": str(path),
        "ukuran_mb": round(path.stat().st_size / 1e6, 2),
        "bentuk_masukan": list(contoh.shape),
        "bentuk_keluaran": list(hasil.shape),
        "selisih_maks": float(np.abs(hasil - acuan).max()),
        "ms_ort_cpu_satu_kali": round(ms, 1),
        "parameter": int(sum(p.numel() for p in model.parameters())),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--keluaran", required=True)
    ap.add_argument("--stgcn-ckpt")
    ap.add_argument("--stcrppg-ckpt")
    args = ap.parse_args()
    folder = pastikan_di_luar_repo(args.keluaran)
    torch.manual_seed(0)
    lap = {"lingkungan": info_lingkungan(), "opset": OPSET}
    m, x = model_stgcn(args.stgcn_ckpt)
    lap["stgcn"] = ekspor("stgcn_va", _KeluaranVA(m), x, folder)
    m, x = model_stcrppg(args.stcrppg_ckpt)
    lap["stcrppg"] = ekspor("stcrppg", m, x, folder)
    tulis_json(folder / "v5_laporan.json", lap)
    print(lap)


if __name__ == "__main__":
    main()
