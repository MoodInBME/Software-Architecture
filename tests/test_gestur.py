"""Modul gestur dengan deret titik sendi sintetis (tanpa video, tanpa mediapipe)."""

import json

import numpy as np
import pytest

from communication.contracts import GestureEvent
from interaction import gesture_features as gf
from interaction.gesture_state import (
    MEMILIH,
    MENUNJUK,
    DIAM,
    MesinGestur,
    PenomorPeristiwa,
    SetelanGestur,
)
from interaction.one_euro import OneEuro

# ---------------------------------------------------------------- tangan sintetis
# Satuan tangan: pergelangan di (0, 0), pangkal jari tengah di (0, 1), sumbu y ke atas.
_MCP = {5: (-0.3, 1.0), 9: (0.0, 1.0), 13: (0.3, 0.95), 17: (0.55, 0.85)}


def _jari(mcp, lurus):
    x, y = _MCP[mcp]
    if lurus:
        return [(x, y), (x, y + 0.45), (x, y + 0.75), (x, y + 1.0)]
    return [(x, y), (x, y + 0.35), (x + 0.02, y + 0.15), (x + 0.02, y - 0.05)]


def tangan(
    jari=(True, True, True, True), ibu_terbuka=True, cubit=None, pusat=(0.5, 0.6), skala=0.12
):
    """Titik (21, 2) koordinat gambar [0, 1]. cubit = rasio jarak ujung ibu jari-telunjuk."""
    pts = [(0.0, 0.0)]
    if ibu_terbuka:
        pts += [(-0.35, 0.25), (-0.6, 0.4), (-0.85, 0.5), (-1.15, 0.6)]
    else:
        pts += [(-0.3, 0.25), (-0.35, 0.45), (-0.25, 0.55), (-0.1, 0.6)]
    for k, mcp in enumerate((5, 9, 13, 17)):
        pts += _jari(mcp, jari[k])
    a = np.asarray(pts, dtype=np.float64)
    if cubit is not None:
        a[6:9] = [(-0.35, 1.4), (-0.5, 1.55), (-0.65, 1.45)]  # telunjuk menekuk ke depan
        a[4] = a[8] + np.array([-cubit, 0.0])
    img = np.empty_like(a)
    img[:, 0] = pusat[0] + skala * a[:, 0]
    img[:, 1] = pusat[1] - skala * a[:, 1]
    return img


TELAPAK = tangan()
TUNJUK = tangan(jari=(True, False, False, False), ibu_terbuka=False)


# ---------------------------------------------------------------- fitur
def test_fitur_tak_bergantung_posisi_dan_skala():
    a = tangan(pusat=(0.3, 0.5), skala=0.1)
    b = tangan(pusat=(0.7, 0.4), skala=0.25)
    np.testing.assert_allclose(gf.normalisasi_tangan(a), gf.normalisasi_tangan(b), atol=1e-9)
    assert gf.ukuran_telapak(b) == pytest.approx(0.25)
    assert gf.rasio_cubit(tangan(cubit=0.1)) == pytest.approx(0.1)


def test_telapak_dan_menunjuk_dibedakan():
    assert gf.telapak_terbuka(TELAPAK)
    assert not gf.telapak_terbuka(TUNJUK)
    assert gf.jari_lurus(TUNJUK).tolist() == [True, False, False, False]
    assert gf.posisi_kursor(TELAPAK) == pytest.approx((0.5 - 0.12 * 0.3, 0.6 - 0.12))
    with pytest.raises(ValueError):
        gf.normalisasi_tangan(np.zeros((20, 2)))


# ---------------------------------------------------------------- One Euro
def test_one_euro_meredam_getar_dan_mengikuti_langkah():
    rng = np.random.default_rng(0)
    f = OneEuro(min_cutoff=1.0, beta=0.007)
    t = np.arange(300) / 30.0
    x = 0.5 + rng.normal(0, 0.01, t.size)
    y = np.array([f(v, s) for v, s in zip(x, t)])
    assert y[100:].std() < 0.5 * x[100:].std()
    f.reset()
    langkah = [f(0.2 if s < 1 else 0.8, s) for s in t]
    assert abs(langkah[-1] - 0.8) < 0.01  # tidak tertinggal setelah ~9 detik
    terakhir = f(0.9, t[-1] + 0.05)
    assert f.n_waktu_tidak_naik == 0
    # Cap waktu sama atau mundur: nilai terakhir dikembalikan, keadaan tidak berubah.
    assert f(0.1, t[-1] + 0.05) == terakhir
    assert f(0.1, t[-1]) == terakhir
    assert f.n_waktu_tidak_naik == 2
    assert f(0.9, t[-1] + 0.1) != terakhir  # waktu naik lagi: filter berjalan normal


# ---------------------------------------------------------------- mesin keadaan
def _jalankan(mesin, urutan, fps=30.0, t0=0.0):
    out, t = [], t0
    for lm in urutan:
        out += mesin.langkah(lm, 0.95 if lm is not None else 0.0, t)
        t += 1.0 / fps
    return out, t


def _jenis(evs):
    return [e.gesture for e in evs if e.gesture != "pointer"]


def test_cubit_satu_select_dan_histeresis():
    m = MesinGestur(SetelanGestur(ambang_tutup=0.30, ambang_buka=0.45))
    rasio = [0.8] * 5 + [0.2] * 3 + [0.33, 0.42, 0.35, 0.40] * 5 + [0.6] * 3 + [0.2] * 5
    urutan = [tangan(jari=(True, False, False, False), ibu_terbuka=False, cubit=r) for r in rasio]
    evs, _ = _jalankan(m, urutan)
    # Getar di antara dua ambang tidak memicu select ulang; buka lalu tutup lagi = select kedua.
    assert _jenis(evs) == ["select", "select"]
    assert m.keadaan == MEMILIH


def test_jeda_pilih_mencegah_select_ganda_cepat():
    m = MesinGestur(SetelanGestur(jeda_pilih_s=0.3))
    rasio = [0.2, 0.6, 0.2, 0.6, 0.2]  # buka-tutup dalam 5 frame (~0,13 s)
    urutan = [tangan(cubit=r, ibu_terbuka=False, jari=(True, False, False, False)) for r in rasio]
    evs, _ = _jalankan(m, urutan)
    assert _jenis(evs) == ["select"]


def test_telapak_ditahan_satu_detik_untuk_kembali():
    m = MesinGestur()
    evs, t = _jalankan(m, [TELAPAK] * 28)  # ~0,93 s
    assert _jenis(evs) == []
    evs2, t = _jalankan(m, [TELAPAK] * 40, t0=t)  # lewat 1 s, tetap ditahan
    assert _jenis(evs2) == ["back"]
    evs3, t = _jalankan(m, [TUNJUK] * 3 + [TELAPAK] * 35, t0=t)  # dilepas lalu ditahan lagi
    assert _jenis(evs3) == ["back"]


def test_tangan_hilang_dan_fase_ukur():
    m = MesinGestur(SetelanGestur(hilang_s=0.3))
    evs, t = _jalankan(m, [TUNJUK] * 5 + [None] * 12)
    assert _jenis(evs) == ["hand_lost"] and m.keadaan == DIAM
    m.set_fase_ukur(True)
    evs, t = _jalankan(m, [TELAPAK] * 60 + [tangan(cubit=0.1)] * 10, t0=t)
    assert evs == [] and m.keadaan == DIAM
    m.set_fase_ukur(False)
    evs, _ = _jalankan(m, [TUNJUK] * 3, t0=t)
    assert evs and m.keadaan == MENUNJUK


def test_pointer_paling_banyak_30_hz_dan_dicerminkan():
    m = MesinGestur(SetelanGestur(maks_hz=30.0))
    evs, t = _jalankan(m, [TUNJUK] * 120, fps=60.0)  # 2 detik pada 60 fps
    ptr = [e for e in evs if e.gesture == "pointer"]
    assert 55 <= len(ptr) <= 61
    # Tangan di kiri gambar -> kursor di kanan layar bila dicerminkan.
    kiri = tangan(jari=(True, False, False, False), ibu_terbuka=False, pusat=(0.3, 0.45))
    m1 = MesinGestur(SetelanGestur(cermin=True))
    m2 = MesinGestur(SetelanGestur(cermin=False))
    x1 = m1.langkah(kiri, 0.9, 0.0)[-1].x
    x2 = m2.langkah(kiri, 0.9, 0.0)[-1].x
    assert x1 > 0.5 > x2


def test_skor_rendah_dianggap_tanpa_tangan():
    m = MesinGestur(SetelanGestur(skor_min=0.5))
    assert m.langkah(TUNJUK, 0.2, 0.0) == []


def test_peristiwa_sesuai_kontrak_dan_seq_naik():
    m = MesinGestur()
    nomor = PenomorPeristiwa("2026-11-22T13-40-00_a1b2")
    rasio = [0.8] * 3 + [0.1] * 3
    urutan = [tangan(cubit=r, ibu_terbuka=False, jari=(True, False, False, False)) for r in rasio]
    urutan += [None] * 15
    evs, _ = _jalankan(m, urutan)
    pesan = [nomor(e) for e in evs]
    assert all(isinstance(p, GestureEvent) for p in pesan)
    assert [p.seq for p in pesan] == list(range(len(pesan)))
    jenis = [p.gesture for p in pesan]
    assert "select" in jenis and jenis[-1] == "hand_lost"
    d = json.loads(pesan[-1].to_json())
    assert d["x"] is None and d["y"] is None and "landmarks" not in d


def test_setelan_tidak_valid():
    with pytest.raises(ValueError):
        SetelanGestur(ambang_tutup=0.5, ambang_buka=0.4)
    with pytest.raises(ValueError):
        SetelanGestur(area_x=(0.8, 0.2))


def test_kotak_sam_demo():
    from scripts.demo_gestur import kotak_di, kotak_sam

    k = kotak_sam(1280, 720)
    assert len(k) == 9 and all(x0 < x1 and y0 < y1 for x0, y0, x1, y1 in k)
    x0, y0, x1, y1 = k[4]
    assert kotak_di((x0 + x1) / 2 / 1280, (y0 + y1) / 2 / 720, k, 1280, 720) == 4
    assert kotak_di(0.5, 0.1, k, 1280, 720) is None


def test_mesin_tahan_cap_waktu_beresolusi_kasar_windows():
    # time.monotonic di Windows berlompatan sekitar 15,6 ms; pada 30 fps dua frame
    # berurutan bisa mendapat cap waktu sama. Mesin tidak boleh berhenti dengan galat.
    m = MesinGestur()
    t_asli = np.arange(90) / 30.0
    t_kasar = np.floor(t_asli / 0.0156) * 0.0156
    t_kasar[10:13] = t_kasar[10]  # tiga frame dengan cap waktu sama
    evs = []
    for k, t in enumerate(t_kasar):
        evs += m.langkah(TUNJUK, 0.9, float(t))
    assert m.filter.n_waktu_tidak_naik >= 2
    assert any(e.gesture == "pointer" for e in evs)


# ---------------------------------------------------------------- zoom (rancangan) dan demo
def _cubit_di(y_pusat, r):
    return tangan(jari=(True, False, False, False), ibu_terbuka=False, cubit=r, pusat=(0.5, y_pusat))


def test_zoom_bawaan_mati_dan_perilaku_select_tidak_berubah():
    assert SetelanGestur().zoom_aktif is False
    m = MesinGestur()
    evs, _ = _jalankan(m, [_cubit_di(0.6, 0.8)] * 3 + [_cubit_di(0.6, 0.1)] * 3)
    assert _jenis(evs) == ["select"]  # select saat cubit menutup


def test_zoom_rancangan_geser_vertikal_dan_tap():
    from interaction.gesture_state import GESTURE_ZOOM

    m = MesinGestur(SetelanGestur(zoom_aktif=True))
    # Cubit tanpa geser lalu lepas: select saat dilepas.
    evs, t = _jalankan(m, [_cubit_di(0.6, 0.8)] * 3 + [_cubit_di(0.6, 0.1)] * 5 + [_cubit_di(0.6, 0.8)] * 2)
    assert _jenis(evs) == ["select"]
    # Cubit lalu geser ke atas: zoom membesar, tidak ada select saat dilepas.
    urutan = [_cubit_di(0.6, 0.1)] * 3 + [_cubit_di(0.6 - 0.02 * k, 0.1) for k in range(1, 15)]
    urutan += [_cubit_di(0.32, 0.8)] * 2
    evs, _ = _jalankan(m, urutan, t0=t + 1.0)
    jenis = _jenis(evs)
    assert "select" not in jenis and jenis.count(GESTURE_ZOOM) >= 3
    zoom = [e.nilai for e in evs if e.gesture == GESTURE_ZOOM]
    assert zoom[-1] > 1.0 and zoom == sorted(zoom)


def test_zoom_tidak_bisa_dikirim_lewat_kontrak():
    from interaction.gesture_state import GESTURE_ZOOM, PeristiwaGestur

    with pytest.raises(ValueError, match="kontrak"):
        PenomorPeristiwa("s")(PeristiwaGestur(GESTURE_ZOOM, 0.5, 0.5, 0.9, 0.0, nilai=1.2))


def test_demo_menggambar_tanpa_kamera_dan_tanpa_menyimpan():
    cv2 = pytest.importorskip("cv2")
    from pathlib import Path

    from scripts import demo_gestur as d

    teks = Path(d.__file__).read_text(encoding="utf-8")
    assert "imwrite" not in teks and "VideoWriter(" not in teks
    m = MesinGestur(SetelanGestur(cermin=False, zoom_aktif=True))
    frame = np.zeros((360, 640, 3), np.uint8)
    for k, lm in enumerate([TUNJUK] * 5 + [TELAPAK] * 20 + [None] * 15):
        m.langkah(lm, 0.9 if lm is not None else 0.0, k / 30.0)
        t = k / 30.0
        aktif = d.gestur_aktif(m, False, t)
        assert aktif in {"tunjuk", "cubit", "telapak", "hilang"}
        kotak = d.kotak_sam(640, 360)
        d._gambar_kamera(cv2, frame, kotak, None, 2, m, False)
        kanvas = np.zeros((360, 640 + d.LEBAR_PANEL, 3), np.uint8)
        d._panel(cv2, kanvas, 640, aktif, d.progres_telapak(m, t), 30.0, ["x"], "Kembali", 1.0)
    assert d.gestur_aktif(m, True, 0.0) == "ukur"
