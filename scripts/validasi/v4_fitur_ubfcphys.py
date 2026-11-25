"""V4 tahap fitur: tabel fitur HRV per jendela untuk 20 subjek UBFC-Phys terpilih.

Sumber sinyal pulsa:
  * ``bvp``  BVP kontak 64 Hz, batas atas (tanpa galat rPPG);
  * ``rppg`` sinyal CHROM dari ``rppg/sN_Tk.npz`` hasil ``v4_unduh_ubfcphys video``.

Per sumber dan panjang jendela (``--jendela 60 120``), jendela digeser setengah panjangnya.
Fitur domain waktu selalu dihitung; LF, HF, LF/HF hanya untuk jendela >= 120 detik
(``SetelanHRV.min_frekuensi_s``).

Normalisasi baseline (``--baseline``):
  * ``tanpa``: fitur mentah.
  * ``t1_awal``: fitur baseline dihitung dari T1 detik 0 sampai ``--baseline-s``; jendela T1
    yang dinilai harus mulai setelah segmen itu, agar jendela uji tidak ikut membentuk
    baseline-nya sendiri. Kolom ``<fitur>_d`` = fitur - baseline.

Akibat panjang tugas UBFC-Phys 180 detik: dengan jendela 120 detik dan baseline 60 detik, T1
hanya menyisakan satu jendela (detik 60-180) per subjek, sedangkan T2 dan T3 masing-masing
dua, sehingga kelas Normal tiga kali lebih sedikit dari tiap kelas stres. Baseline 60 detik
juga terlalu pendek untuk LF dan HF, jadi kolom ``lf_d``, ``hf_d``, ``lf_hf_d`` berisi NaN;
pakai fitur frekuensi mentah atau ``--baseline-s 120`` (yang mengosongkan T1). Jumlah jendela
per kelas dilaporkan agar hal ini terlihat sebelum pelatihan.

    python -m scripts.validasi.v4_fitur_ubfcphys --akar D:/moodin_runs/ubfc_phys ^
        --keluaran D:/moodin_runs/v4 --jendela 60 120 --baseline t1_awal --baseline-s 60
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data.loaders.ubfc_phys import (
    FS_BVP,
    TUGAS,
    KELAS_STRES,
    baca_bvp,
    jendela,
    label_tugas,
)
from models.stress.hrv import FITUR_FREKUENSI, FITUR_WAKTU, SetelanHRV, fitur_jendela
from scripts.validasi._umum import info_lingkungan, pastikan_di_luar_repo, tulis_json


def sinyal_subjek(
    akar: Path, sid: str, sumber: str
) -> dict[str, tuple[np.ndarray, float]]:
    out = {}
    for t in TUGAS:
        if sumber == "bvp":
            out[t] = (baca_bvp(akar / "sinyal" / sid / f"bvp_{sid}_{t}.csv"), FS_BVP)
        else:
            p = akar / "rppg" / f"{sid}_{t}.npz"
            if not p.exists():
                raise FileNotFoundError(f"{p} belum ada; jalankan tahap video dulu.")
            z = np.load(p)
            out[t] = (z["rppg"].astype(np.float64), float(z["fps"]))
    return out


def baris_subjek(sid, kelompok, sinyal, panjang_s, setelan, baseline, baseline_s):
    fitur = setelan.fitur
    dasar = None
    mulai_t1 = 0.0
    if baseline == "t1_awal":
        x, fs = sinyal["T1"]
        seg = x[: int(round(baseline_s * fs))]
        s_dasar = SetelanHRV(
            **{
                **setelan.__dict__,
                "fitur": tuple(f for f in fitur if f in FITUR_WAKTU)
                if baseline_s < setelan.min_frekuensi_s
                else fitur,
            }
        )
        dasar = fitur_jendela(seg, fs, s_dasar)
        mulai_t1 = baseline_s
    out = []
    for t in TUGAS:
        x, fs = sinyal[t]
        for a, b in jendela(
            len(x), fs, panjang_s, panjang_s / 2, mulai_t1 if t == "T1" else 0.0
        ):
            f = fitur_jendela(x[a:b], fs, setelan)
            r = {
                "subjek": sid,
                "kelompok": kelompok,
                "tugas": t,
                "label": label_tugas(t, kelompok),
                "mulai_s": round(a / fs, 2),
                "panjang_s": panjang_s,
                "valid": f is not None,
            }
            if f is not None:
                r.update({k: f[k] for k in fitur})
                r.update(
                    n_ibi=f["n_ibi"],
                    proporsi_ibi_valid=round(f["proporsi_ibi_valid"], 3),
                )
                if baseline == "t1_awal":
                    for k in fitur:
                        r[f"{k}_d"] = (
                            (f[k] - dasar[k])
                            if dasar is not None and k in dasar
                            else float("nan")
                        )
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--akar", required=True)
    ap.add_argument("--keluaran", required=True)
    ap.add_argument("--sumber", nargs="+", default=["bvp"], choices=("bvp", "rppg"))
    ap.add_argument("--jendela", nargs="+", type=float, default=[60.0, 120.0])
    ap.add_argument("--baseline", choices=("tanpa", "t1_awal"), default="t1_awal")
    ap.add_argument("--baseline-s", type=float, default=60.0)
    ap.add_argument(
        "--metode-frekuensi", choices=("welch", "lombscargle"), default="welch"
    )
    args = ap.parse_args()

    akar = Path(args.akar)
    folder = pastikan_di_luar_repo(args.keluaran)
    pil = json.loads((akar / "pilihan.json").read_text(encoding="utf-8"))["pilihan"]
    lap = {
        "lingkungan": info_lingkungan(),
        "subjek": pil,
        "baseline": args.baseline,
        "baseline_s": args.baseline_s,
        "pemetaan_kelas": "ASUMSI: T1 Normal; T2/T3 ctrl Stres Ringan; T2/T3 test Stres Akut",
        "tabel": {},
    }
    for sumber in args.sumber:
        for panjang in args.jendela:
            setelan = SetelanHRV(
                metode_frekuensi=args.metode_frekuensi,
                fitur=FITUR_WAKTU
                + (FITUR_FREKUENSI if panjang >= SetelanHRV.min_frekuensi_s else ()),
            )
            semua = []
            for sid, kel in pil.items():
                semua += baris_subjek(
                    sid,
                    kel,
                    sinyal_subjek(akar, sid, sumber),
                    panjang,
                    setelan,
                    args.baseline,
                    args.baseline_s,
                )
            nama = f"fitur_{sumber}_{int(panjang)}s_{args.baseline}.csv"
            kolom = sorted(
                {k for r in semua for k in r},
                key=lambda k: (
                    k
                    not in (
                        "subjek",
                        "kelompok",
                        "tugas",
                        "label",
                        "mulai_s",
                        "panjang_s",
                        "valid",
                    ),
                    k,
                ),
            )
            with open(folder / nama, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=kolom)
                w.writeheader()
                w.writerows(semua)
            valid = [r for r in semua if r["valid"]]
            per_kelas = {
                KELAS_STRES[k]: sum(1 for r in valid if r["label"] == k)
                for k in range(3)
            }
            lap["tabel"][nama] = {
                "jendela": len(semua),
                "valid": len(valid),
                "per_kelas": per_kelas,
                "fitur": list(setelan.fitur),
            }
            print(nama, lap["tabel"][nama])
            if 0 in [per_kelas[k] for k in KELAS_STRES]:
                print(
                    f"  PERINGATAN: ada kelas kosong di {nama}; LOSO tiga kelas tidak bisa dinilai adil."
                )
    tulis_json(folder / "v4_fitur_ringkasan.json", lap)


if __name__ == "__main__":
    main()
