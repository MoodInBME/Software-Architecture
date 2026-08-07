#!/usr/bin/env python3
"""MoodIn - Diagnostik kegagalan warmup di Jetson.

Pesan yang muncul di log Anda:
    Warmup GAGAL: too many values to unpack (expected 4)

Artinya ada baris pembongkaran tuple dengan EMPAT nama di sisi kiri,
sementara sisi kanan menghasilkan LEBIH dari empat nilai. Pada ST-GCN,
tersangka utamanya adalah baris semacam:

    N, C, T, V = x.size()

yang menerima tensor lima dimensi (N, C, T, V, M).

Skrip ini TIDAK menebak. Ia mencoba beberapa bentuk tensor satu per satu
terhadap model yang sesungguhnya, lalu mencetak traceback lengkap yang
selama ini ditelan oleh blok try/except di inference.py.

Jalankan dari akar repo:
    cd ~/MoodIn_Software && python3 diag_warmup.py 2>&1 | tee /tmp/diag_warmup.log
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402


def garis(judul: str) -> None:
    print()
    print("=" * 70)
    print(judul)
    print("=" * 70)


def cari_pembongkaran_empat() -> None:
    """Menelusuri sumber untuk baris pembongkaran empat nama."""
    garis("1. Baris kandidat penyebab 'expected 4' di dalam sumber")
    import re

    pola = re.compile(r"^\s*([A-Za-z_]\w*\s*,\s*){3}[A-Za-z_]\w*\s*=\s*.*"
                      r"(\.size\(\)|\.shape|\.size\(0\)|unpack)")
    akar = Path(__file__).resolve().parent
    ketemu = 0
    for berkas in sorted(akar.rglob("*.py")):
        if "venv" in berkas.parts or berkas.name == Path(__file__).name:
            continue
        try:
            baris = berkas.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, isi in enumerate(baris, 1):
            if pola.match(isi):
                ketemu += 1
                print(f"   {berkas.relative_to(akar)}:{i}")
                print(f"       {isi.strip()}")
    if ketemu == 0:
        print("   (tidak ada baris yang cocok; berarti sumbernya di tempat lain)")
    else:
        print(f"   total {ketemu} baris kandidat")


def muat_model():
    """Memuat kedua model persis seperti inference.py melakukannya."""
    garis("2. Memuat model")
    stgcn = None
    rppg = None

    try:
        from models.st_gcn_graph_tran.model import STGCN_GraphTransformer
        stgcn = STGCN_GraphTransformer(
            in_channels=2,
            num_class=5,
            num_node=68,
            graph_strategy="spatial",
            max_hop=1,
            t_kernel=9,
            transformer_layers=2,
            transformer_heads=8,
            transformer_ff=512,
            dropout=0.3,
            head_type="va",
        )
        stgcn.eval()
        print("   ST-GCN  : termuat")
    except Exception:
        print("   ST-GCN  : GAGAL dimuat")
        traceback.print_exc()

    try:
        from models.stc_rppg.model import STCrPPG
        rppg = STCrPPG()
        rppg.eval()
        print("   STC-rPPG: termuat")
    except Exception:
        print("   STC-rPPG: GAGAL dimuat")
        traceback.print_exc()

    return stgcn, rppg


def uji_bentuk(nama_model, model, bentuk_list) -> None:
    garis(f"3. Uji bentuk tensor untuk {nama_model}")
    if model is None:
        print("   model tidak tersedia, dilewati")
        return

    for bentuk in bentuk_list:
        x = torch.zeros(*bentuk)
        label = " x ".join(str(d) for d in bentuk)
        try:
            with torch.no_grad():
                keluaran = model(x)
        except Exception as exc:  # noqa: BLE001
            print(f"   [GAGAL] {label:<28} -> {type(exc).__name__}: {exc}")
            # Cetak hanya dua bingkai terakhir agar terbaca, dengan nomor baris.
            tb = traceback.extract_tb(exc.__traceback__)
            for bingkai in tb[-3:]:
                print(f"            {bingkai.filename}:{bingkai.lineno} "
                      f"dalam {bingkai.name}")
                if bingkai.line:
                    print(f"                {bingkai.line.strip()}")
            continue

        if isinstance(keluaran, (tuple, list)):
            ringkas = ", ".join(
                str(tuple(o.shape)) if hasattr(o, "shape") else type(o).__name__
                for o in keluaran
            )
            print(f"   [BISA]  {label:<28} -> ({ringkas})")
        elif hasattr(keluaran, "shape"):
            print(f"   [BISA]  {label:<28} -> {tuple(keluaran.shape)}")
        else:
            print(f"   [BISA]  {label:<28} -> {type(keluaran).__name__}")


def main() -> int:
    garis("0. Lingkungan")
    print("   python :", sys.version.split()[0])
    print("   torch  :", torch.__version__)
    print("   cuda   :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("   gpu    :", torch.cuda.get_device_name(0))

    cari_pembongkaran_empat()
    stgcn, rppg = muat_model()

    # ST-GCN: 150 bingkai, 68 titik, 2 kanal (x dan y).
    # Varian empat dimensi dan lima dimensi diuji berdampingan supaya
    # terlihat mana yang sebenarnya diterima oleh forward().
    uji_bentuk(
        "ST-GCN",
        stgcn,
        [
            (1, 2, 150, 68),        # N C T V
            (1, 2, 150, 68, 1),     # N C T V M
            (1, 150, 68, 2),        # N T V C
            (1, 2, 32, 68),         # klip pendek, N C T V
        ],
    )

    # STC-rPPG: 128 bingkai RGB 128x128.
    uji_bentuk(
        "STC-rPPG",
        rppg,
        [
            (1, 3, 128, 128, 128),  # N C T H W
            (1, 3, 128, 64, 64),
            (1, 128, 3, 128, 128),  # N T C H W
            (1, 3, 32, 32, 32),     # bentuk warmup 32 pangkat tiga
        ],
    )

    garis("SELESAI")
    print("Kirimkan seluruh isi /tmp/diag_warmup.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
