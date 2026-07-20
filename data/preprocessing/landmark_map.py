"""Pemetaan MediaPipe FaceMesh (468 titik) -> subset iBUG-68 (dlib).

Dipakai agar keluaran FaceMesh kompatibel dengan topologi graf 68-titik di
`models/st_gcn_graph_tran/graph.py`. Urutan indeks mengikuti konvensi iBUG-68
(jaw 0-16, alis 17-26, hidung 27-35, mata 36-47, bibir 48-67).

CATATAN [Medium]: indeks ini perlu diverifikasi sekali dengan overlay visual
(gambar wajah + titik) karena pemetaan 468->68 bersifat aproksimasi.
"""
from __future__ import annotations

import numpy as np

# 68 indeks MediaPipe FaceMesh yang mendekati titik iBUG-68 (berurutan iBUG).
MEDIAPIPE_FACEMESH_TO_IBUG68 = [
    # Jaw (0-16)
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    # Alis kanan (17-21) & kiri (22-26)
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
    # Batang hidung (27-30) & bawah hidung (31-35)
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    # Mata kanan (36-41) & kiri (42-47)
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    # Bibir luar (48-59) & dalam (60-67)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
]

assert len(MEDIAPIPE_FACEMESH_TO_IBUG68) == 68, "Pemetaan harus tepat 68 titik"

# Rentang region pada indeks iBUG-68 (untuk analisis/plot per-region).
IBUG68_REGIONS = {
    "jaw": list(range(0, 17)),
    "eyebrow_right": list(range(17, 22)),
    "eyebrow_left": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_bottom": list(range(31, 36)),
    "eye_right": list(range(36, 42)),
    "eye_left": list(range(42, 48)),
    "mouth_outer": list(range(48, 60)),
    "mouth_inner": list(range(60, 68)),
}

_INDEX = np.asarray(MEDIAPIPE_FACEMESH_TO_IBUG68, dtype=np.int64)


def subset_to_ibug68(landmarks: np.ndarray) -> np.ndarray:
    """Ambil 68 titik iBUG dari array FaceMesh.

    landmarks: (..., 468, D) -> keluaran (..., 68, D). Mengambil pada sumbu -2.
    """
    landmarks = np.asarray(landmarks)
    if landmarks.shape[-2] < 468:
        raise ValueError(f"Diharapkan >=468 titik, dapat {landmarks.shape[-2]}")
    return np.take(landmarks, _INDEX, axis=-2)


if __name__ == "__main__":
    dummy = np.random.rand(150, 468, 2)
    out = subset_to_ibug68(dummy)
    print("in", dummy.shape, "-> out", out.shape)  # (150, 68, 2)
