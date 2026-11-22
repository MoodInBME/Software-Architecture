"""V5: data dan cache kalibrasi INT8 untuk ST-GCN.

Dua langkah:
  * ``buat`` (laptop): ambil N jendela landmark AFEW-VA ternormalisasi dari manifest
    hasil ``afew_va.py prepare`` dan simpan sebagai satu .npy (N, 2, 90, 68) float32 di
    luar repo. Isinya titik landmark, bukan citra.
  * ``kalibrasi`` (Jetson, TensorRT Python): buat cache kalibrasi entropi dari .npy itu
    untuk ``trtexec --int8 --calib=...``.

    python -m scripts.validasi.v5_kalibrasi_int8 buat --manifest D:/prepared/afew-va/manifest.json --keluaran D:/moodin_runs/v5/kalib_stgcn.npy --n 512
    python3 -m scripts.validasi.v5_kalibrasi_int8 kalibrasi --onnx ~/moodin_runs/v5/stgcn_va.onnx --data ~/moodin_runs/v5/kalib_stgcn.npy --cache ~/moodin_runs/v5/stgcn_int8.cache

Catatan: kalibrator implisit TensorRT ditandai usang sejak TensorRT 10, tetapi masih
tersedia di versi JetPack 6 [Sedang, cek pesan peringatan saat dijalankan].
"""

from __future__ import annotations

import argparse

import numpy as np


def buat(manifest: str, keluaran: str, n: int, seed: int = 0) -> np.ndarray:
    from data.loaders.affect_landmark_dataset import AffectLandmarkDataset
    from scripts.validasi._umum import pastikan_di_luar_repo

    out = pastikan_di_luar_repo(keluaran)
    ds = AffectLandmarkDataset(manifest, au_list=[], clip_len=90)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    x = np.stack([ds[int(i)][0].numpy() for i in idx]).astype(np.float32)
    np.save(out, x)
    print(f"{x.shape} -> {out}")
    return x


def kalibrasi(onnx_path: str, data: str, cache: str, batch: int = 8) -> None:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    x = np.load(data).astype(np.float32)

    class Kalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.i = 0
            self.dev = cuda.mem_alloc(x[:batch].nbytes)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= len(x):
                return None
            cuda.memcpy_htod(self.dev, np.ascontiguousarray(x[self.i : self.i + 1]))
            self.i += 1
            return [int(self.dev)]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, c):
            with open(cache, "wb") as f:
                f.write(c)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(net, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise SystemExit(
                "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            )
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.int8_calibrator = Kalibrator()
    if builder.build_serialized_network(net, cfg) is None:
        raise SystemExit("Pembangunan engine INT8 gagal")
    print(f"cache kalibrasi -> {cache}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("buat")
    b.add_argument("--manifest", required=True)
    b.add_argument("--keluaran", required=True)
    b.add_argument("--n", type=int, default=512)
    k = sub.add_parser("kalibrasi")
    k.add_argument("--onnx", required=True)
    k.add_argument("--data", required=True)
    k.add_argument("--cache", required=True)
    a = ap.parse_args()
    if a.cmd == "buat":
        buat(a.manifest, a.keluaran, a.n)
    else:
        kalibrasi(a.onnx, a.data, a.cache)


if __name__ == "__main__":
    main()
