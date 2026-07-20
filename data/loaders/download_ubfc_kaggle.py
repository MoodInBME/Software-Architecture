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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="malekdinarito/ubfc-rppg-dataset",
                    help="slug dataset Kaggle 'owner/nama'")
    ap.add_argument("--path", default="dataset", help="folder tujuan")
    ap.add_argument("--no-unzip", action="store_true",
                    help="jangan otomatis meng-unzip")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()  # membaca ~/.kaggle/kaggle.json atau env vars
    api.dataset_download_files(args.slug, path=args.path, unzip=not args.no_unzip)
    print(f"Selesai mengunduh {args.slug} -> {args.path}")


if __name__ == "__main__":
    main()
