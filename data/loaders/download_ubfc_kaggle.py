"""Unduh UBFC-rPPG dari Kaggle TANPA menaruh token/rahasia di dalam kode.

Pustaka Kaggle membaca kredensial secara otomatis dari (urutan prioritas):
  1. ~/.kaggle/kaggle.json   (chmod 600)
  2. env var KAGGLE_USERNAME + KAGGLE_KEY

JANGAN pernah menuliskan API token langsung di file yang di-commit. Jika sebuah
token sudah pernah ter-commit, CABUT/rotasi token itu di akun Kaggle Anda.

Contoh:
    export KAGGLE_USERNAME=... ; export KAGGLE_KEY=...
    python download_ubfc_kaggle.py --path dataset
"""
import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="malekdinarito/ubfc-rppg-dataset")
    ap.add_argument("--path", default="dataset")
    ap.add_argument("--no-unzip", action="store_true")
    args = ap.parse_args()

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.slug,
        "-p",
        args.path,
    ]

    if not args.no_unzip:
        cmd.append("--unzip")

    print("Memulai download...")
    subprocess.run(cmd, check=True)
    print("Download selesai!")


if __name__ == "__main__":
    main()
