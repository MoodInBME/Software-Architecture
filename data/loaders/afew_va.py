"""AFEW-VA (Kossaifi dkk., Image and Vision Computing 2017) -> klip landmark berlabel VA.

Sumber resmi: https://ibug.doc.ic.ac.uk/resources/afew-va-database/
Lisensi: hanya untuk riset. Wajib menyitir Kossaifi dkk. 2017 dan Dhall dkk. 2012
(IEEE MultiMedia 19(3)). Data TIDAK boleh masuk repo; simpan di luar folder repo.

Tata letak yang diharapkan setelah 12 zip diekstrak [Medium, cek satu folder dulu]:
    <root>/<klip>/<klip>.json   # {"actor": ..., "video_id": ..., "frames": {"00000": {...}}}
    <root>/<klip>/00000.png ...
Setiap entri frame berisi "valence", "arousal" (bilangan bulat, rentang -10..10 menurut
makalah) dan "landmarks" (68 titik [x, y]). Bila struktur berbeda, `read_clip_json`
melempar galat yang menyebut kunci yang hilang, bukan diam-diam menghasilkan data salah.

Contoh:
    python data/loaders/afew_va.py download --dest ../datasets/afew-va --saya-setuju-riset
    python data/loaders/afew_va.py prepare --root ../datasets/afew-va \
        --output_dir ../prepared/afew-va --landmarks extract --clip_len 90 --stride 15
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.preprocessing.affect_windows import (  # noqa: E402
    fill_missing,
    make_clips,
    normalize_landmarks_seq,
    resample_time,
    save_clips,
)

BASE_URL = "https://ibug.doc.ic.ac.uk/download/AFEW-VA/"
ZIP_NAMES = [f"{i:02d}.zip" for i in range(1, 13)]
VA_SCALE = 10.0  # -10..10 -> -1..1 [Medium: periksa min dan maks label setelah unduh]
# Laju frame klip AFEW belum dinyatakan di halaman dataset. Ukur dari video sumber atau
# tetapkan lewat --src_fps setelah diperiksa. Nilai ini hanya bawaan sementara.
DEFAULT_SRC_FPS = 25.0


# ----------------------------------------------------------------------------- unduh
def _inside(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def download(dest: str, agree: bool, timeout: int = 60) -> list[dict]:
    """Unduh 12 zip ke `dest` (di luar repo), lewati yang sudah ada, catat SHA-256."""
    if not agree:
        raise SystemExit(
            "AFEW-VA hanya untuk riset. Jalankan ulang dengan --saya-setuju-riset "
            "setelah membaca ketentuan di halaman dataset."
        )
    d = Path(dest)
    if _inside(d, _ROOT):
        raise SystemExit(
            f"Folder tujuan {d} berada di dalam repo. Pilih folder di luar repo."
        )
    d.mkdir(parents=True, exist_ok=True)
    log = []
    for name in ZIP_NAMES:
        target = d / name
        if not target.exists():
            tmp = target.with_suffix(".part")
            print(f"Mengunduh {name} ...")
            try:
                with (
                    urllib.request.urlopen(BASE_URL + name, timeout=timeout) as r,
                    open(tmp, "wb") as f,
                ):
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            except OSError as exc:
                if tmp.exists():
                    tmp.unlink()
                raise SystemExit(f"Gagal mengunduh {name}: {exc}") from exc
            tmp.rename(target)
        h = hashlib.sha256()
        with open(target, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        log.append(
            {"file": name, "bytes": target.stat().st_size, "sha256": h.hexdigest()}
        )
    with open(d / "sha256_afew_va.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print(f"Selesai. Ekstrak zip secara manual ke {d}, lalu jalankan 'prepare'.")
    return log


# ----------------------------------------------------------------------------- baca
def read_clip_json(path: str) -> dict:
    """Baca satu JSON klip -> dict berisi frame terurut, VA (T, 2) ternormalisasi, landmark GT."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "frames" not in data or not isinstance(data["frames"], dict):
        raise KeyError(f"{path}: kunci 'frames' tidak ada; periksa struktur AFEW-VA")
    keys = sorted(data["frames"].keys(), key=lambda k: int(k))
    va, lms = [], []
    for k in keys:
        fr = data["frames"][k]
        missing = [x for x in ("valence", "arousal", "landmarks") if x not in fr]
        if missing:
            raise KeyError(f"{path} frame {k}: kunci {missing} tidak ada")
        va.append([float(fr["valence"]), float(fr["arousal"])])
        pts = np.asarray(fr["landmarks"], dtype=np.float32)
        if pts.shape != (68, 2):
            raise ValueError(
                f"{path} frame {k}: landmark {pts.shape}, diharapkan (68, 2)"
            )
        lms.append(pts)
    va = np.asarray(va, dtype=np.float32)
    if np.abs(va).max(initial=0) > VA_SCALE:
        raise ValueError(f"{path}: |VA| melebihi {VA_SCALE}; periksa skala label")
    clip_id = Path(path).stem
    return {
        "clip": clip_id,
        "actor": str(data.get("actor", clip_id)),
        "frame_keys": keys,
        "va": va / VA_SCALE,
        "landmarks_gt": np.stack(lms, axis=0),
        "dir": str(Path(path).parent),
    }


def iter_clip_jsons(root: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
    return [p for p in paths if not p.endswith("sha256_afew_va.json")]


def _frame_path(clip_dir: str, key: str) -> str:
    for ext in (".png", ".jpg"):
        p = os.path.join(clip_dir, key + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Frame {key} tidak ditemukan di {clip_dir}")


def extract_landmarks(clip: dict, extractor) -> np.ndarray:
    """Landmark dari ekstraktor yang sama dengan cermin (MediaPipe -> iBUG-68).

    Return (T, 68, 2) ternormalisasi dengan NaN pada frame gagal deteksi.
    """
    import cv2

    out = []
    for key in clip["frame_keys"]:
        img = cv2.imread(_frame_path(clip["dir"], key))
        if img is None:
            out.append(np.full((68, 2), np.nan, np.float32))
            continue
        pts = extractor.extract(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out.append(np.full((68, 2), np.nan, np.float32) if pts is None else pts)
    return np.stack(out, axis=0)


def prepare(
    root,
    output_dir,
    landmarks,
    clip_len,
    stride,
    src_fps,
    dst_fps,
    max_filled,
    extractor=None,
):
    manifest, skipped = [], []
    clips_dir = os.path.join(output_dir, "clips")
    for jp in iter_clip_jsons(root):
        clip = read_clip_json(jp)
        if landmarks == "gt":
            seq = normalize_landmarks_seq(clip["landmarks_gt"])
        else:
            seq = extract_landmarks(clip, extractor)
        try:
            seq, valid = fill_missing(seq)
        except ValueError:
            skipped.append(clip["clip"])
            continue
        if src_fps != dst_fps:
            seq = resample_time(seq, src_fps, dst_fps)
            valid = resample_time(valid.astype(np.float32), src_fps, dst_fps) > 0.999
            va = resample_time(clip["va"], src_fps, dst_fps)
        else:
            va = clip["va"]
        recs = make_clips(
            seq,
            valid,
            f"afewva_{clip['clip']}",
            clip["actor"],
            "afew-va",
            clip_len,
            stride,
            va=va,
        )
        recs = [r for r in recs if r.frac_filled <= max_filled]
        save_clips(recs, clips_dir, manifest)
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"AFEW-VA: {len(manifest)} klip, {len(skipped)} klip tanpa wajah dilewati")
    return manifest


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--dest", required=True)
    d.add_argument("--saya-setuju-riset", action="store_true", dest="agree")
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--landmarks",
        choices=["extract", "gt"],
        default="extract",
        help="extract = MediaPipe seperti di cermin (bawaan, menghindari pergeseran domain); gt = anotasi",
    )
    p.add_argument("--clip_len", type=int, default=90)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--src_fps", type=float, default=DEFAULT_SRC_FPS)
    p.add_argument("--dst_fps", type=float, default=30.0)
    p.add_argument(
        "--max_filled",
        type=float,
        default=0.2,
        help="batas fraksi frame isian per klip",
    )
    args = ap.parse_args()
    if args.cmd == "download":
        download(args.dest, args.agree)
        return
    if _inside(Path(args.output_dir), _ROOT):
        raise SystemExit(
            "output_dir di dalam repo. Klip turunan juga data dataset, simpan di luar repo."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    extractor = None
    if args.landmarks == "extract":
        from data.preprocessing.landmark_extraction_mediapipe import (
            MediaPipeLandmarkExtractor,
        )

        extractor = MediaPipeLandmarkExtractor(static_image_mode=False)
    try:
        prepare(
            args.root,
            args.output_dir,
            args.landmarks,
            args.clip_len,
            args.stride,
            args.src_fps,
            args.dst_fps,
            args.max_filled,
            extractor,
        )
    finally:
        if extractor is not None:
            extractor.close()


if __name__ == "__main__":
    main()
