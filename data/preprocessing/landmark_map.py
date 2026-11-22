"""Pemetaan MediaPipe FaceMesh (468 titik) -> subset iBUG-68: SATU sumber untuk repo.

Dipakai agar keluaran FaceMesh kompatibel dengan topologi graf 68-titik di
`models/st_gcn_graph_tran/graph.py`. Urutan indeks mengikuti konvensi iBUG-68
(jaw 0-16, alis 17-26, hidung 27-35, mata 36-47, bibir 48-67).

Ada dua varian yang berbeda tepat di titik iBUG 0, 17, dan 26:
  * ``A``: tabel lama berkas ini, dipakai ``inference.py`` (runtime cermin).
  * ``B``: tabel lama ``landmark_extraction_mediapipe.py``, dipakai ekstraksi dataset
    (``extract_landmarks_video.py``, ``afew_va.py``, ``disfa.py``).
Varian belum dipilih: keputusan menunggu uji NME pada 300-W
(``scripts/validasi/v1_nme_300w.py``). Sampai saat itu setiap pemanggil memakai
varian yang sama seperti sebelumnya, sehingga runtime dan data latih masih memakai
tabel yang berbeda di tiga titik itu (dicatat sebagai isu di log lintas repo).

Kedua tabel adalah aproksimasi 468 -> 68, bukan angka dari paper.
"""
from __future__ import annotations

import numpy as np

_TABEL_A = [
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

# Varian B = varian A kecuali tiga titik berikut (indeks iBUG -> indeks FaceMesh).
_SELISIH_B = {0: 127, 17: 70, 26: 300}
_TABEL_B = [_SELISIH_B.get(i, v) for i, v in enumerate(_TABEL_A)]

VARIAN_PETA = {"A": tuple(_TABEL_A), "B": tuple(_TABEL_B)}
TITIK_BERBEDA = tuple(sorted(_SELISIH_B))

for _nama, _tabel in VARIAN_PETA.items():
    assert len(_tabel) == 68, f"Peta {_nama} harus tepat 68 titik"
    assert len(set(_tabel)) == 68, f"Peta {_nama} memuat indeks ganda"
    assert all(0 <= i < 468 for i in _tabel), f"Peta {_nama} di luar 0..467"

# Nama lama tetap ada supaya pemanggil tidak berubah perilaku.
MEDIAPIPE_FACEMESH_TO_IBUG68 = list(VARIAN_PETA["A"])  # inference.py
MEDIAPIPE_TO_IBUG68 = list(VARIAN_PETA["B"])  # ekstraksi dataset


def peta_ibug68(varian: str) -> list[int]:
    """Daftar 68 indeks FaceMesh untuk varian ``A`` atau ``B``."""
    try:
        return list(VARIAN_PETA[varian])
    except KeyError:
        raise ValueError(f"varian peta harus salah satu dari {sorted(VARIAN_PETA)}, bukan {varian!r}") from None


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

def subset_to_ibug68(landmarks: np.ndarray, varian: str = "A") -> np.ndarray:
    """Ambil 68 titik iBUG dari array FaceMesh.

    landmarks: (..., 468, D) -> keluaran (..., 68, D). Mengambil pada sumbu -2.
    """
    landmarks = np.asarray(landmarks)
    if landmarks.shape[-2] < 468:
        raise ValueError(f"Diharapkan >=468 titik, dapat {landmarks.shape[-2]}")
    return np.take(landmarks, np.asarray(peta_ibug68(varian), dtype=np.int64), axis=-2)


if __name__ == "__main__":
    dummy = np.random.rand(150, 468, 2)
    out = subset_to_ibug68(dummy)
    print("in", dummy.shape, "-> out", out.shape)  # (150, 68, 2)
