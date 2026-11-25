"""Daftar AU kepala ST-GCN dan peta indeksnya ke tiap sumber label.

Bimbingan 24 November 2026 menetapkan empat AU: AU1, AU4, AU6, AU12 (AU7 diganti AU1).
Labelnya datang dari dua sumber dengan jenis berbeda:

  * DISFA: intensitas 0..5 per frame, diubah ke 0..1 lewat ``au_scale.ke_satuan``.
  * Aff-Wild2 (ABAW): kemunculan 0/1 per frame; -1 berarti tidak dianotasi.

Urutan kolom di tiap sumber tidak sama dengan urutan kepala, jadi pemetaan selalu lewat
nomor AU, bukan posisi kolom. Kesalahan urutan di sini tidak memunculkan galat saat
pelatihan; model hanya belajar AU yang salah.
"""

from __future__ import annotations

from dataclasses import dataclass

AU_TA: tuple[int, ...] = (1, 4, 6, 12)

INTENSITAS = "intensitas"
KEMUNCULAN = "kemunculan"

# Urutan kolom berkas label DISFA (lihat data/loaders/disfa.py).
DISFA_AUS: tuple[int, ...] = (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26)
# Urutan kolom anotasi AU Aff-Wild2 pada ABAW. Cocokkan dengan baris judul berkas
# anotasi setelah unduh; berkas yang urutannya lain ditolak oleh ``peta_indeks``.
AFFWILD2_AUS: tuple[int, ...] = (1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26)


@dataclass(frozen=True)
class SumberAU:
    nama: str
    jenis: str
    au: tuple[int, ...]


SUMBER = {
    "disfa": SumberAU("disfa", INTENSITAS, DISFA_AUS),
    "affwild2": SumberAU("affwild2", KEMUNCULAN, AFFWILD2_AUS),
}


def peta_indeks(target, sumber_au) -> list[int | None]:
    """Untuk tiap AU di ``target``, posisi kolomnya di ``sumber_au`` atau None bila tidak ada."""
    sumber_au = [int(a) for a in sumber_au]
    if len(set(sumber_au)) != len(sumber_au):
        raise ValueError(f"Daftar AU sumber berisi duplikat: {sumber_au}")
    return [sumber_au.index(int(a)) if int(a) in sumber_au else None for a in target]


def validasi_au_list(
    target, sumber=("disfa", "affwild2")
) -> dict[str, list[int | None]]:
    """Periksa bahwa setiap AU target punya label di minimal satu sumber.

    Mengembalikan peta indeks per sumber. Galat bila ada AU target yang tidak dimiliki
    sumber mana pun, karena kepala AU itu tidak akan pernah mendapat gradien.
    """
    target = [int(a) for a in target]
    if len(set(target)) != len(target):
        raise ValueError(f"au_list berisi duplikat: {target}")
    tak_dikenal = [s for s in sumber if s not in SUMBER]
    if tak_dikenal:
        raise ValueError(
            f"Sumber AU tidak dikenal: {tak_dikenal}. Pilihan: {sorted(SUMBER)}"
        )
    peta = {s: peta_indeks(target, SUMBER[s].au) for s in sumber}
    yatim = [a for j, a in enumerate(target) if all(peta[s][j] is None for s in sumber)]
    if yatim:
        raise ValueError(
            f"AU {yatim} tidak punya label di sumber {list(sumber)}. "
            "Hapus dari au_list atau tambahkan sumber yang memuatnya."
        )
    return peta
