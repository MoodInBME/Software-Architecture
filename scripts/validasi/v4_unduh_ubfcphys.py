"""V4 tahap data: berkas sinyal 56 subjek UBFC-Phys, pilih 10 test + 10 ctrl, lalu video bertahap.

Dijalankan di laptop, bukan di cloud (video UBFC tidak boleh keluar dari komputer lokal).
Kredensial tidak pernah ditulis ke berkas: bila situs pengelola dataset meminta login,
unduh manual lewat peramban lalu pakai ``--sumber-lokal``.

Tiga tahap, masing-masing bisa diulang (berkas yang sudah ada dilewati):

1. ``sinyal``  info, selfReportedAnx, bvp, eda untuk s1..s56 (berukuran kecil).
2. ``pilih``   baca kelompok dari info, cek kelengkapan BVP, pilih 10 test + 10 ctrl dengan
               seed tetap, tulis ``pilihan.json``.
3. ``video``   untuk subjek terpilih, satu video per kali: unduh, ekstraksi rerata RGB ROI
               kulit dan sinyal CHROM (frame tidak disimpan), simpan ``rppg/sN_Tk.npz``,
               lalu hapus video. Video di ``--sumber-lokal`` hanya dihapus bila
               ``--hapus-sumber`` diberikan.

Sumber berkas, pilih salah satu:
  --templat-url "https://.../{sid}/{berkas}"   {sid} = s1.., {berkas} = nama berkas
  --daftar-url daftar.csv                      kolom: berkas,url
  --sumber-lokal D:/unduhan_ubfc               berkas sudah diunduh manual (boleh bersubfolder)

    python -m scripts.validasi.v4_unduh_ubfcphys sinyal --akar D:/moodin_runs/ubfc_phys --sumber-lokal D:/unduhan_ubfc
    python -m scripts.validasi.v4_unduh_ubfcphys pilih --akar D:/moodin_runs/ubfc_phys
    python -m scripts.validasi.v4_unduh_ubfcphys video --akar D:/moodin_runs/ubfc_phys --sumber-lokal D:/unduhan_ubfc
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np

from data.loaders.ubfc_phys import (
    TUGAS,
    baca_kelompok,
    berkas_sinyal,
    pilih_seimbang,
    subjek_lengkap,
)
from scripts.validasi._umum import info_lingkungan, pastikan_di_luar_repo, tulis_json

N_SUBJEK = 56


class Sumber:
    def __init__(self, templat=None, daftar=None, lokal=None):
        if sum(x is not None for x in (templat, daftar, lokal)) != 1:
            raise SystemExit(
                "Pilih tepat satu: --templat-url, --daftar-url, atau --sumber-lokal."
            )
        self.templat, self.lokal = templat, Path(lokal) if lokal else None
        self.daftar = {}
        if daftar:
            with open(daftar, newline="", encoding="utf-8") as f:
                self.daftar = {
                    r["berkas"].strip(): r["url"].strip() for r in csv.DictReader(f)
                }
        self._indeks_lokal = None

    def _cari_lokal(self, berkas: str) -> Path | None:
        if self._indeks_lokal is None:
            self._indeks_lokal = {
                p.name: p for p in self.lokal.rglob("*") if p.is_file()
            }
        return self._indeks_lokal.get(berkas)

    def ambil(self, sid: str, berkas: str, tujuan: Path) -> str:
        """Salin atau unduh berkas ke tujuan. Mengembalikan 'lokal', 'unduh', atau 'tidak_ada'."""
        if self.lokal is not None:
            p = self._cari_lokal(berkas)
            if p is None:
                return "tidak_ada"
            if p.resolve() != tujuan.resolve():
                shutil.copyfile(p, tujuan)
            return "lokal"
        url = (
            self.templat.format(sid=sid, berkas=berkas)
            if self.templat
            else self.daftar.get(berkas)
        )
        if not url:
            return "tidak_ada"
        sementara = tujuan.with_suffix(tujuan.suffix + ".part")
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as r,
                open(sementara, "wb") as f,
            ):
                shutil.copyfileobj(r, f, length=1 << 20)
        except OSError as e:
            sementara.unlink(missing_ok=True)
            print(f"  gagal {berkas}: {e}")
            return "tidak_ada"
        sementara.replace(tujuan)
        return "unduh"


def tahap_sinyal(akar: Path, sumber: Sumber) -> None:
    hitung = {"lokal": 0, "unduh": 0, "tidak_ada": 0, "sudah_ada": 0}
    for i in range(1, N_SUBJEK + 1):
        sid = f"s{i}"
        d = akar / "sinyal" / sid
        d.mkdir(parents=True, exist_ok=True)
        for b in berkas_sinyal(sid):
            t = d / b
            if t.exists() and t.stat().st_size > 0:
                hitung["sudah_ada"] += 1
                continue
            hitung[sumber.ambil(sid, b, t)] += 1
    print("Berkas sinyal:", hitung)


def tahap_pilih(akar: Path, n: int, seed: int) -> dict:
    kelompok, status = {}, {}
    for i in range(1, N_SUBJEK + 1):
        sid = f"s{i}"
        ok, alasan = subjek_lengkap(akar / "sinyal" / sid, sid)
        status[sid] = alasan
        if ok:
            kelompok[sid] = baca_kelompok(akar / "sinyal" / sid / f"info_{sid}.txt")
    pilihan = pilih_seimbang(kelompok, n, seed)
    hasil = {
        "lingkungan": info_lingkungan(),
        "seed": seed,
        "n_per_kelompok": n,
        "jumlah_lengkap": {
            k: sum(1 for g in kelompok.values() if g == k) for k in ("test", "ctrl")
        },
        "pilihan": {s: kelompok[s] for s in pilihan},
        "status_subjek": status,
    }
    tulis_json(akar / "pilihan.json", hasil)
    print("Lengkap:", hasil["jumlah_lengkap"], "Terpilih:", len(pilihan))
    return hasil


def tahap_video(akar: Path, sumber: Sumber, hapus_sumber: bool) -> None:
    from data.preprocessing.chrom import chrom
    from scripts.validasi.v3_chrom_ubfcphys import rgb_streaming

    pil = json.loads((akar / "pilihan.json").read_text(encoding="utf-8"))["pilihan"]
    keluar = akar / "rppg"
    keluar.mkdir(exist_ok=True)
    sementara = akar / "_video_sementara"
    sementara.mkdir(exist_ok=True)
    for sid in pil:
        for t in TUGAS:
            npz = keluar / f"{sid}_{t}.npz"
            if npz.exists():
                continue
            nama = f"vid_{sid}_{t}.avi"
            asal = sumber._cari_lokal(nama) if sumber.lokal is not None else None
            if asal is not None:
                video, dari = asal, "lokal"
            else:
                video = sementara / nama
                dari = sumber.ambil(sid, nama, video)
                if dari == "tidak_ada":
                    print(f"  {nama} tidak tersedia; dilewati")
                    continue
            t0 = time.perf_counter()
            rgb, fps = rgb_streaming(str(video))
            sinyal = chrom(rgb, fps)
            np.savez_compressed(
                npz,
                rgb=rgb.astype(np.float32),
                rppg=sinyal.astype(np.float32),
                fps=fps,
                persen_tanpa_wajah=float(np.isnan(rgb).any(axis=1).mean() * 100),
            )
            print(
                f"  {sid} {t}: {len(rgb)} frame, {fps:.2f} fps, {time.perf_counter() - t0:.0f} s"
            )
            if dari == "unduh" or (dari == "lokal" and hapus_sumber):
                Path(video).unlink()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tahap", choices=("sinyal", "pilih", "video"))
    ap.add_argument("--akar", required=True, help="folder kerja di luar repo")
    ap.add_argument("--templat-url")
    ap.add_argument("--daftar-url")
    ap.add_argument("--sumber-lokal")
    ap.add_argument("--n-per-kelompok", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hapus-sumber", action="store_true")
    args = ap.parse_args()
    akar = pastikan_di_luar_repo(args.akar)
    if args.tahap == "pilih":
        tahap_pilih(akar, args.n_per_kelompok, args.seed)
        return
    sumber = Sumber(args.templat_url, args.daftar_url, args.sumber_lokal)
    if args.tahap == "sinyal":
        tahap_sinyal(akar, sumber)
    else:
        tahap_video(akar, sumber, args.hapus_sumber)


if __name__ == "__main__":
    main()
