"""V4 tahap data dan fitur pada folder UBFC-Phys sintetis (tanpa video)."""

import argparse
import csv
import json
import sys

import numpy as np
import pytest

from data.loaders import ubfc_phys as U
from scripts.validasi import v4_fitur_ubfcphys as F
from scripts.validasi import v4_unduh_ubfcphys as D
from tests.test_hrv import ibi_bermodulasi, pulsa


def _subjek(folder, sid, kelompok, rng, durasi=180, rusak=False):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"info_{sid}.txt").write_text(f"m\n{kelompok}\n2019-01-01\n10:00\n")
    (folder / f"selfReportedAnx_{sid}.csv").write_text("1,2,3\n")
    for t, a_hf in (("T1", 40), ("T2", 15), ("T3", 10)):
        x = pulsa(
            ibi_bermodulasi(durasi + 2, mean_ms=850 if t == "T1" else 700, a_hf=a_hf),
            seed=int(rng.integers(1000)),
        )
        x = (
            x[: int(durasi * U.FS_BVP)]
            if not (rusak and t == "T3")
            else x[: int(30 * U.FS_BVP)]
        )
        np.savetxt(folder / f"bvp_{sid}_{t}.csv", x, delimiter=",")
        np.savetxt(folder / f"eda_{sid}_{t}.csv", np.zeros(10), delimiter=",")


def test_baca_kelompok_dan_label():
    assert U.label_tugas("T1", "test") == 0
    assert U.label_tugas("T2", "ctrl") == 1 and U.label_tugas("T3", "test") == 2
    with pytest.raises(ValueError):
        U.label_tugas("T4", "test")


def test_baca_kelompok_menolak_info_ambigu(tmp_path):
    p = tmp_path / "info.txt"
    p.write_text("F\nTEST\n")
    assert U.baca_kelompok(p) == "test"
    p.write_text("test ctrl\n")
    with pytest.raises(ValueError, match="kelompok"):
        U.baca_kelompok(p)
    p.write_text("contest\n")  # kata 'test' di dalam kata lain tidak dihitung
    with pytest.raises(ValueError):
        U.baca_kelompok(p)


def test_jendela_dan_pilih_seimbang():
    assert U.jendela(int(180 * 64), 64, 60, 30) == [
        (i * 1920, i * 1920 + 3840) for i in range(5)
    ]
    assert len(U.jendela(int(180 * 64), 64, 120, 60, mulai_s=60)) == 1
    kel = {f"s{i}": ("test" if i % 2 else "ctrl") for i in range(1, 31)}
    a = U.pilih_seimbang(kel, 10, seed=3)
    assert a == U.pilih_seimbang(kel, 10, seed=3)
    assert sum(kel[s] == "test" for s in a) == 10 and len(a) == 20
    with pytest.raises(ValueError, match="hanya punya"):
        U.pilih_seimbang(kel, 16, seed=0)


def test_alur_sinyal_pilih_fitur(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    unduhan = tmp_path / "unduhan"
    for i in range(1, 7):
        _subjek(
            unduhan / f"s{i}",
            f"s{i}",
            "test" if i <= 3 else "ctrl",
            rng,
            rusak=(i == 6),
        )
    monkeypatch.setattr(D, "N_SUBJEK", 6)
    akar = tmp_path / "kerja"
    D.tahap_sinyal(akar, D.Sumber(lokal=str(unduhan)))
    assert (akar / "sinyal" / "s1" / "bvp_s1_T2.csv").exists()
    with pytest.raises(ValueError, match="ctrl hanya punya 2"):
        D.tahap_pilih(akar, 3, seed=0)
    hasil = D.tahap_pilih(akar, 2, seed=0)
    assert "30 s" in hasil["status_subjek"]["s6"]
    assert sorted(hasil["pilihan"].values()) == ["ctrl", "ctrl", "test", "test"]

    keluar = tmp_path / "v4"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v4",
            "--akar",
            str(akar),
            "--keluaran",
            str(keluar),
            "--jendela",
            "60",
            "120",
            "--baseline",
            "t1_awal",
        ],
    )
    F.main()
    lap = json.loads((keluar / "v4_fitur_ringkasan.json").read_text(encoding="utf-8"))
    t60 = lap["tabel"]["fitur_bvp_60s_t1_awal.csv"]
    assert (
        t60["per_kelas"]["Normal"] == 3 * 4
    )  # T1 sesudah baseline 60 s: 3 jendela x 4 subjek
    assert "lf_hf" not in t60["fitur"]
    t120 = lap["tabel"]["fitur_bvp_120s_t1_awal.csv"]
    assert "lf_hf" in t120["fitur"]
    # T1 180 s: baseline 0-60 s menyisakan tepat satu jendela 120 s (60-180 s) per subjek.
    assert t120["per_kelas"]["Normal"] == 4 and t120["per_kelas"]["Stres Akut"] == 8
    with open(keluar / "fitur_bvp_60s_t1_awal.csv", encoding="utf-8") as f:
        baris = [r for r in csv.DictReader(f) if r["valid"] == "True"]
    t1 = [float(r["mean_nn_d"]) for r in baris if r["tugas"] == "T1"]
    t2 = [float(r["mean_nn_d"]) for r in baris if r["tugas"] == "T2"]
    assert (
        abs(np.mean(t1)) < 30 and np.mean(t2) < -100
    )  # T2 lebih cepat dari baseline T1


def test_sumber_harus_tepat_satu():
    with pytest.raises(SystemExit):
        D.Sumber()
    with pytest.raises(SystemExit):
        D.Sumber(templat="x/{sid}/{berkas}", lokal=".")
