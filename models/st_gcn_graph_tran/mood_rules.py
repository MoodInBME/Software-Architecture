"""Pilihan aturan pemetaan valensi-arousal ke lima kategori mood.

* ``tabel33`` (bawaan): aturan rancangan Pra-TA di ``labels_tabel33.py``.
* ``lingkaran``: aturan era GEMASTIK di ``labels.py`` (zona Netral berbentuk
  lingkaran), dipertahankan untuk arsip dan perbandingan.

Dua mode pemetaan (tanpa keadaan, untuk evaluasi):

* ``kind="label"``: label acuan (misalnya rating SAM). Kombinasi yang belum
  tercakup aturan ``tabel33`` diberi ``MOOD_GAP_ID`` (-1); nilai ini diabaikan rugi
  klasifikasi dan dikeluarkan dari metrik kategori, dan jumlahnya dilaporkan.
* ``kind="prediction"``: keluaran model kontinu. Titik dengan v tepat 0 di luar
  pita Netral mengikuti konvensi v >= 0 sebagai sisi positif, sehingga setiap
  prediksi selalu mendapat kategori.

``decision_margin`` adalah heuristik jarak (metrik Chebyshev) ke batas kategori
terdekat, bukan peluang terkalibrasi. ``MoodDisplayHysteresis`` hanya untuk tampilan
cermin dan tidak dipakai saat evaluasi. Modul ini tidak bergantung pada torch.
"""

from __future__ import annotations

from .labels import MOOD_TO_ID, va_to_mood_id
from .labels_tabel33 import BAND, va_to_mood_id_tabel33

MOOD_RULES = ("tabel33", "lingkaran")
MOOD_GAP_ID = -1


MOOD_KINDS = ("label", "prediction")


def va_to_mood_prediction(v: float, a: float, band: float = BAND) -> str:
    """Kategori untuk keluaran model. v tepat 0 di luar pita Netral dianggap sisi positif."""
    if abs(v) <= band and abs(a) <= band:
        return "Netral"
    if v >= 0:
        return "Ceria" if a > 0 else "Rileks"
    return "Cemas" if a > 0 else "Tertekan"


def is_rule_gap(v: float, a: float, band: float = BAND) -> bool:
    """True bila (v, a) jatuh di celah aturan: v tepat 0 di luar pita Netral."""
    return v == 0 and not (abs(v) <= band and abs(a) <= band)


def mood_id_function(rule: str = "tabel33", neutral_radius: float = 0.2,
                     kind: str = "label"):
    """Return fungsi (v, a) -> id mood 0..4, atau MOOD_GAP_ID untuk celah label."""
    rule = str(rule).lower()
    if kind not in MOOD_KINDS:
        raise ValueError(f"kind harus salah satu dari {MOOD_KINDS}, bukan {kind!r}")
    if rule == "tabel33" and kind == "prediction":
        def fn(v, a):
            return MOOD_TO_ID[va_to_mood_prediction(float(v), float(a))]
        return fn
    if rule == "tabel33":

        def fn(v, a):
            m = va_to_mood_id_tabel33(float(v), float(a))
            return MOOD_GAP_ID if m is None else m

        return fn
    if rule == "lingkaran":

        def fn(v, a):
            return va_to_mood_id(float(v), float(a), neutral_radius)

        return fn
    raise ValueError(f"mood_rule harus salah satu dari {MOOD_RULES}, bukan {rule!r}")


def boundary_distance(v: float, a: float, band: float = BAND) -> float:
    """Jarak Chebyshev (v, a) ke batas kategori terdekat.

    Di dalam pita Netral: band - max(|v|, |a|). Di luar pita: minimum dari jarak ke
    pita Netral, jarak ke garis v = 0, dan jarak ke garis a = 0, dengan jarak ke
    garis hanya dihitung pada bagian garis yang berada di luar pita.
    """
    av, aa = abs(float(v)), abs(float(a))
    m = max(av, aa)
    if m <= band:
        return band - m
    ke_pita = m - band
    ke_garis_v = av if aa >= band else max(av, band - aa)
    ke_garis_a = aa if av >= band else max(aa, band - av)
    return min(ke_pita, ke_garis_v, ke_garis_a)


def decision_margin(v: float, a: float, face_conf: float = 1.0,
                    margin_scale: float = BAND, band: float = BAND) -> float:
    """Skor margin keputusan: face_conf x (0,5 + 0,5 x min(1, d / margin_scale)).

    Heuristik jarak ke batas aturan, bukan peluang. Bernilai 0,5 x face_conf tepat
    di batas dan face_conf bila d >= margin_scale.
    """
    if margin_scale <= 0:
        raise ValueError("margin_scale harus positif")
    d = boundary_distance(v, a, band)
    m = float(face_conf) * (0.5 + 0.5 * min(1.0, d / margin_scale))
    return max(0.0, min(1.0, m))


class MoodDisplayHysteresis:
    """Histeresis kategori untuk tampilan cermin (bukan untuk evaluasi).

    Kategori yang sedang tampil dipertahankan selama masih menjadi kandidat, yaitu
    kategori dari salah satu titik dalam bola Chebyshev berjari-jari ``width`` di
    sekitar (v, a). Selain itu kategori prediksi biasa dipakai.
    """

    def __init__(self, width: float = 0.05, band: float = BAND):
        if width < 0:
            raise ValueError("width harus >= 0")
        self.width = float(width)
        self.band = float(band)
        self.current = None

    def candidates(self, v: float, a: float) -> set:
        h = self.width
        offsets = (-h, 0.0, h) if h > 0 else (0.0,)
        out = {va_to_mood_prediction(v + dv, a + da, self.band)
               for dv in offsets for da in offsets}
        if is_rule_gap(v, a, self.band):
            out |= {va_to_mood_prediction(-1e-12, a, self.band)}
        return out

    def update(self, v: float, a: float) -> str:
        cand = self.candidates(v, a)
        if self.current not in cand:
            self.current = va_to_mood_prediction(v, a, self.band)
        return self.current

    def reset(self):
        self.current = None


class MoodRuleDataset:
    """Bungkus dataset (x, va, mood) dan hitung ulang mood dari va dengan aturan pilihan.

    Dipakai agar ``ExpressionLandmarkDataset`` (aturan lingkaran bawaan) tidak perlu
    diubah. Label VA tidak disentuh. Kelas ini sengaja tidak mewarisi
    ``torch.utils.data.Dataset`` agar modul tetap bebas torch; DataLoader dan
    random_split cukup membutuhkan ``__len__`` dan ``__getitem__``.
    """

    def __init__(self, base, mood_fn):
        self.base = base
        self.mood_fn = mood_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        import torch

        x, va, _ = self.base[idx]
        mood = torch.tensor(self.mood_fn(va[0], va[1]), dtype=torch.long)
        return x, va, mood
