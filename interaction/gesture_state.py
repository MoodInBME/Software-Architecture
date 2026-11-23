"""Mesin keadaan gestur: diam, menunjuk, memilih; telapak ditahan untuk kembali.

Masukan per frame: titik sendi tangan (21, 2) dalam koordinat gambar [0, 1] atau
None bila tidak ada tangan, skor deteksi, dan cap waktu monoton (detik). Keluaran:
daftar peristiwa dengan jenis sesuai kontrak IPC 2.0.0 (``pointer``, ``select``,
``back``, ``hand_lost``). Peristiwa diubah menjadi ``GestureEvent`` oleh pemanggil.

Aturan:
  * Kursor dari pangkal telunjuk (titik 5), dipetakan dari area aktif kamera ke
    [0, 1], dicerminkan bila ``cermin``, lalu diredam filter One Euro.
  * Cubit memakai histeresis dua ambang: menutup bila rasio < ``ambang_tutup``,
    membuka bila rasio > ``ambang_buka``. ``select`` dikirim sekali saat menutup.
  * Telapak terbuka yang ditahan ``tahan_kembali_s`` mengirim ``back`` sekali; telapak
    harus dilepas dulu sebelum ``back`` berikutnya.
  * Selama fase ukur tidak ada peristiwa sama sekali dan keadaan kembali ke diam.
  * ``pointer`` dibatasi ``maks_hz`` per detik.

Nilai ambang bawaan adalah titik awal, bukan hasil kalibrasi; kalibrasi dengan
``scripts/demo_gestur.py`` (angka rasio cubit ditampilkan di layar).

Zoom (RANCANGAN, bawaan mati, menunggu keputusan Calon Dosen Pembimbing 1): bila
``zoom_aktif``, ``select`` dikirim saat cubit DILEPAS tanpa gerakan, sedangkan cubit
yang ditahan lalu digeser vertikal lebih dari ``zoom_ambang`` (dalam satuan layar)
menghasilkan peristiwa ``zoom`` dengan ``nilai`` = faktor skala relatif terhadap awal
cubit (geser ke atas memperbesar). ``zoom`` belum ada di kontrak IPC 2.0.0, sehingga
``PenomorPeristiwa`` menolaknya; peristiwa ini hanya dipakai lokal di demo.
"""

from __future__ import annotations

from dataclasses import dataclass

from communication import topics
from interaction.gesture_features import posisi_kursor, rasio_cubit, telapak_terbuka
from interaction.one_euro import OneEuro2D

GESTURE_ZOOM = "zoom"  # rancangan; bukan bagian kontrak IPC 2.0.0

DIAM = "diam"
MENUNJUK = "menunjuk"
MEMILIH = "memilih"


@dataclass
class PeristiwaGestur:
    gesture: str
    x: float | None
    y: float | None
    score: float
    t: float
    nilai: float | None = None  # faktor zoom (hanya untuk GESTURE_ZOOM)


@dataclass
class SetelanGestur:
    ambang_tutup: float = 0.30
    ambang_buka: float = 0.45
    tahan_kembali_s: float = 1.0
    hilang_s: float = 0.3
    jeda_pilih_s: float = 0.3
    maks_hz: float = topics.GESTURE_POINTER_MAX_HZ
    skor_min: float = 0.5
    cermin: bool = True
    area_x: tuple[float, float] = (0.2, 0.8)
    area_y: tuple[float, float] = (0.15, 0.75)
    min_cutoff: float = 1.0
    beta: float = 0.007
    zoom_aktif: bool = False  # rancangan, menunggu keputusan bimbingan
    zoom_ambang: float = 0.04  # geser vertikal minimum (satuan layar) sebelum dianggap zoom
    zoom_kepekaan: float = 2.0  # faktor = 1 + kepekaan * (y_awal - y)

    def __post_init__(self):
        if not 0 < self.ambang_tutup < self.ambang_buka:
            raise ValueError("Harus 0 < ambang_tutup < ambang_buka (histeresis)")
        for lo, hi in (self.area_x, self.area_y):
            if not 0 <= lo < hi <= 1:
                raise ValueError("Area aktif harus di dalam [0, 1] dengan lo < hi")
        if self.maks_hz <= 0 or self.tahan_kembali_s <= 0:
            raise ValueError("maks_hz dan tahan_kembali_s harus > 0")


def _peta(v: float, lo: float, hi: float) -> float:
    return min(1.0, max(0.0, (v - lo) / (hi - lo)))


class MesinGestur:
    def __init__(self, setelan: SetelanGestur | None = None):
        self.s = setelan or SetelanGestur()
        self.filter = OneEuro2D(min_cutoff=self.s.min_cutoff, beta=self.s.beta)
        self.fase_ukur = False
        self._reset()

    def _reset(self) -> None:
        self.keadaan = DIAM
        self.cubit = False
        self.rasio = None
        self.kursor = None
        self._t_terakhir_tangan = None
        self._t_pointer = None
        self._t_pilih = None
        self._t_palm = None
        self._palm_terkirim = False
        self._cubit_y0 = None
        self._sedang_zoom = False
        self.faktor_zoom = 1.0
        self.filter.reset()

    def set_fase_ukur(self, aktif: bool) -> None:
        """True selama fase pengukuran: gestur nonaktif dan keadaan direset."""
        self.fase_ukur = bool(aktif)
        if self.fase_ukur:
            self._reset()

    def _cubit_baru(self, rasio: float) -> bool:
        if self.cubit:
            if rasio > self.s.ambang_buka:
                self.cubit = False
        elif rasio < self.s.ambang_tutup:
            self.cubit = True
        return self.cubit

    def langkah(self, lm, skor: float, t: float) -> list[PeristiwaGestur]:
        if self.fase_ukur:
            return []
        out: list[PeristiwaGestur] = []
        ada = lm is not None and skor >= self.s.skor_min
        if not ada:
            if (
                self.keadaan != DIAM
                and self._t_terakhir_tangan is not None
                and t - self._t_terakhir_tangan > self.s.hilang_s
            ):
                self._reset()
                out.append(PeristiwaGestur(topics.GESTURE_HAND_LOST, None, None, 0.0, t))
            return out

        self._t_terakhir_tangan = t
        cx, cy = posisi_kursor(lm)
        if self.s.cermin:
            cx = 1.0 - cx
        x = _peta(cx, *self.s.area_x)
        y = _peta(cy, *self.s.area_y)
        x, y = self.filter(x, y, t)
        x, y = min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
        self.kursor = (x, y)
        skor = float(min(1.0, max(0.0, skor)))

        # Kembali: telapak terbuka ditahan; tidak dihitung saat mencubit.
        if telapak_terbuka(lm) and not self.cubit:
            if self._t_palm is None:
                self._t_palm = t
            elif not self._palm_terkirim and t - self._t_palm >= self.s.tahan_kembali_s:
                self._palm_terkirim = True
                out.append(PeristiwaGestur(topics.GESTURE_BACK, None, None, skor, t))
        else:
            self._t_palm = None
            self._palm_terkirim = False

        self.rasio = rasio_cubit(lm)
        sebelum = self.cubit
        sekarang = self._cubit_baru(self.rasio)
        if self.s.zoom_aktif:
            out += self._cubit_dengan_zoom(sebelum, sekarang, x, y, skor, t)
        elif sekarang and not sebelum:
            out += self._pilih(x, y, skor, t)
        self.keadaan = MEMILIH if sekarang else MENUNJUK

        if self._t_pointer is None or t - self._t_pointer >= 1.0 / self.s.maks_hz - 1e-9:
            self._t_pointer = t
            out.append(PeristiwaGestur(topics.GESTURE_POINTER, x, y, skor, t))
        return out


    def _pilih(self, x, y, skor, t) -> list[PeristiwaGestur]:
        if self._t_pilih is not None and t - self._t_pilih < self.s.jeda_pilih_s:
            return []
        self._t_pilih = t
        return [PeristiwaGestur(topics.GESTURE_SELECT, x, y, skor, t)]

    def _cubit_dengan_zoom(self, sebelum, sekarang, x, y, skor, t) -> list[PeristiwaGestur]:
        """Rancangan zoom: pilih saat cubit dilepas tanpa geser; geser vertikal = zoom."""
        if sekarang and not sebelum:
            self._cubit_y0, self._sedang_zoom = y, False
            return []
        if sekarang and self._cubit_y0 is not None:
            dy = self._cubit_y0 - y
            if self._sedang_zoom or abs(dy) >= self.s.zoom_ambang:
                self._sedang_zoom = True
                self.faktor_zoom = max(0.25, 1.0 + self.s.zoom_kepekaan * dy)
                return [PeristiwaGestur(GESTURE_ZOOM, x, y, skor, t, nilai=self.faktor_zoom)]
            return []
        if sebelum and not sekarang:
            zoom = self._sedang_zoom
            self._cubit_y0, self._sedang_zoom = None, False
            return [] if zoom else self._pilih(x, y, skor, t)
        return []


class PenomorPeristiwa:
    """Mengubah PeristiwaGestur menjadi GestureEvent kontrak 2.0.0 dengan ``seq`` naik."""

    def __init__(self, session_id: str, mulai: int = 0):
        self.session_id = session_id
        self.seq = int(mulai)

    def __call__(self, p: PeristiwaGestur):
        from communication.contracts import GestureEvent

        if p.gesture not in topics.GESTURES:
            raise ValueError(f"Peristiwa {p.gesture!r} tidak ada di kontrak IPC 2.0.0; tidak dikirim lewat MQTT")
        ev = GestureEvent(
            session_id=self.session_id,
            seq=self.seq,
            gesture=p.gesture,
            x=None if p.x is None else round(p.x, 4),
            y=None if p.y is None else round(p.y, 4),
            score=round(p.score, 3),
        ).validate()
        self.seq += 1
        return ev
