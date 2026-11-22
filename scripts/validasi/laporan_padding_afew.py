"""Laporan padding jendela AFEW-VA: berapa jendela terbentuk dan berapa persen frame padding.

Hanya membaca JSON anotasi AFEW-VA (jumlah frame per klip); tidak membuka gambar.
Hasil ditulis ke JSON di luar repo.

    python -m scripts.validasi.laporan_padding_afew --root D:/afew-va --keluaran D:/moodin_runs/v2/padding_afew.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.preprocessing.affect_windows import chunk_indices
from scripts.validasi._umum import pastikan_di_luar_repo


def ringkas_padding(panjang, clip_len: int = 90, stride: int = 15) -> dict:
    """Ringkasan jendela untuk daftar panjang klip (frame).

    Klip yang lebih pendek dari ``clip_len`` menghasilkan satu jendela yang dipad
    sampai ``clip_len`` (pengulangan frame terakhir, ``AffectLandmarkDataset``).
    """
    panjang = [int(n) for n in panjang]
    if not panjang:
        raise ValueError("daftar panjang klip kosong")
    jendela = 0
    jendela_pad = 0
    frame_pad = 0
    for n in panjang:
        idx = chunk_indices(n, clip_len, stride)
        jendela += len(idx)
        for s, e in idx:
            kurang = clip_len - (e - s)
            if kurang > 0:
                jendela_pad += 1
                frame_pad += kurang
    total_frame_jendela = jendela * clip_len
    p = np.asarray(panjang)
    return {
        "klip": len(panjang),
        "klip_pendek": int((p < clip_len).sum()),
        "panjang_min": int(p.min()),
        "panjang_median": float(np.median(p)),
        "panjang_maks": int(p.max()),
        "clip_len": clip_len,
        "stride": stride,
        "jendela": jendela,
        "jendela_berpadding": jendela_pad,
        "frame_padding": frame_pad,
        "persen_frame_padding": round(100.0 * frame_pad / total_frame_jendela, 2),
    }


def panjang_dari_root(root: str) -> list[int]:
    from data.loaders.afew_va import iter_clip_jsons

    out = []
    for jp in iter_clip_jsons(root):
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        if "frames" not in data:
            continue
        out.append(len(data["frames"]))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--keluaran", required=True)
    ap.add_argument("--clip_len", type=int, nargs="+", default=[90, 48])
    ap.add_argument("--stride", type=int, default=15)
    args = ap.parse_args()
    out = pastikan_di_luar_repo(args.keluaran)
    panjang = panjang_dari_root(args.root)
    hasil = [ringkas_padding(panjang, c, args.stride) for c in args.clip_len]
    out.write_text(json.dumps(hasil, indent=1), encoding="utf-8")
    for h in hasil:
        print(json.dumps(h, ensure_ascii=False))


if __name__ == "__main__":
    main()
