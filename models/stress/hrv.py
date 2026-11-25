"""Fitur HRV dari sinyal pulsa (BVP kontak atau rPPG): puncak, IBI, domain waktu, domain frekuensi.

Urutan: sinyal difilter pada pita denyut (0,75-4 Hz, sama dengan jalur rPPG), puncak
dicari dengan jarak minimum sesuai HR maksimum, IBI dihitung dari selisih waktu puncak,
lalu IBI yang tidak fisiologis dibuang sebelum fitur dihitung.

Fitur domain waktu: MeanNN, SDNN, RMSSD (ms) dan MeanHR (bpm).
Fitur domain frekuensi: daya LF (0,04-0,15 Hz), HF (0,15-0,4 Hz) dalam ms^2, dan LF/HF.
Satu siklus LF terlemah (0,04 Hz) berlangsung 25 detik, sehingga jendela pendek tidak
memuat cukup siklus untuk estimasi yang stabil. Fitur frekuensi karena itu hanya dihitung
bila jendela minimal ``min_frekuensi_s`` (bawaan 120 detik); jendela lebih pendek ditolak,
bukan diisi nilai yang tampak sah. Durasi fase ukur belum diputuskan (bimbingan 24 November
2026), jadi pilihan 60 atau 120 detik ada di config, bukan di kode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from models.stc_rppg.constants import HR_BAND_HZ

FITUR_WAKTU = ("mean_nn", "sdnn", "rmssd", "mean_hr")
FITUR_FREKUENSI = ("lf", "hf", "lf_hf")
PITA_LF = (0.04, 0.15)
PITA_HF = (0.15, 0.40)


@dataclass
class SetelanHRV:
    ibi_min_ms: float = 333.0  # 180 bpm
    ibi_maks_ms: float = 1500.0  # 40 bpm
    # IBI yang berbeda lebih dari ambang ini terhadap median lokal dianggap puncak salah
    # atau puncak terlewat; ambang 20 persen umum dipakai untuk penyaring berbasis median.
    ambang_lokal: float = 0.20
    lebar_median: int = 5
    min_frekuensi_s: float = 120.0
    metode_frekuensi: str = "welch"  # welch | lombscargle
    fs_interpolasi: float = 4.0
    min_ibi_valid: float = 0.8  # proporsi IBI tersisa minimal agar jendela dipakai
    fitur: tuple[str, ...] = field(default=FITUR_WAKTU)

    def __post_init__(self):
        if self.metode_frekuensi not in ("welch", "lombscargle"):
            raise ValueError("metode_frekuensi harus 'welch' atau 'lombscargle'")
        tak_dikenal = [f for f in self.fitur if f not in FITUR_WAKTU + FITUR_FREKUENSI]
        if tak_dikenal:
            raise ValueError(f"Fitur tidak dikenal: {tak_dikenal}")


def filter_pulsa(x: np.ndarray, fs: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [HR_BAND_HZ[0], HR_BAND_HZ[1]], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def cari_puncak(x: np.ndarray, fs: float) -> np.ndarray:
    """Indeks puncak sistolik pada sinyal yang sudah difilter."""
    from scipy.signal import find_peaks

    jarak = max(1, int(round(fs / HR_BAND_HZ[1])))
    # Prominence relatif terhadap sebaran sinyal agar tidak bergantung satuan BVP/rPPG.
    p, _ = find_peaks(x, distance=jarak, prominence=0.3 * float(np.std(x)))
    return p


def ibi_dari_puncak(puncak: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(waktu akhir tiap IBI dalam detik, IBI dalam ms)."""
    t = np.asarray(puncak, dtype=np.float64) / fs
    return t[1:], np.diff(t) * 1000.0


def saring_ibi(
    t: np.ndarray, ibi: np.ndarray, s: SetelanHRV
) -> tuple[np.ndarray, np.ndarray]:
    """Buang IBI di luar rentang fisiologis atau menyimpang dari median lokal."""
    ok = (ibi >= s.ibi_min_ms) & (ibi <= s.ibi_maks_ms)
    t, ibi = t[ok], ibi[ok]
    if ibi.size < 3:
        return t, ibi
    h = s.lebar_median // 2
    med = np.array([np.median(ibi[max(0, i - h) : i + h + 1]) for i in range(ibi.size)])
    ok = np.abs(ibi - med) <= s.ambang_lokal * med
    return t[ok], ibi[ok]


def fitur_waktu(ibi: np.ndarray) -> dict:
    d = np.diff(ibi)
    return {
        "mean_nn": float(ibi.mean()),
        "sdnn": float(ibi.std(ddof=1)),
        # RMSSD dari selisih berurutan setelah penyaringan; IBI yang dibuang di tengah
        # membuat dua IBI tak bersebelahan ikut diselisihkan, ditoleransi bila jarang.
        "rmssd": float(np.sqrt(np.mean(d**2))),
        "mean_hr": float(np.mean(60000.0 / ibi)),
    }


def _daya_pita(f: np.ndarray, p: np.ndarray, pita) -> float:
    m = (f >= pita[0]) & (f < pita[1])
    return float(np.trapz(p[m], f[m])) if m.sum() > 1 else 0.0


def fitur_frekuensi(
    t: np.ndarray, ibi: np.ndarray, durasi_s: float, s: SetelanHRV
) -> dict:
    if durasi_s < s.min_frekuensi_s:
        raise ValueError(
            f"Jendela {durasi_s:g} s lebih pendek dari {s.min_frekuensi_s:g} s; LF/HF tidak dihitung. "
            "Pakai jendela lebih panjang atau keluarkan fitur frekuensi dari config."
        )
    if s.metode_frekuensi == "welch":
        from scipy.interpolate import CubicSpline
        from scipy.signal import welch

        tg = np.arange(t[0], t[-1], 1.0 / s.fs_interpolasi)
        y = CubicSpline(t, ibi)(tg)
        y = y - y.mean()
        nper = min(len(y), int(64 * s.fs_interpolasi))
        f, p = welch(
            y, fs=s.fs_interpolasi, nperseg=nper, noverlap=nper // 2, detrend="linear"
        )
    else:
        from scipy.signal import lombscargle

        f = np.linspace(0.01, 0.5, 491)
        y = ibi - ibi.mean()
        # Lomb-Scargle bekerja langsung pada IBI yang tidak berjarak sama (tanpa interpolasi).
        # Periodogramnya tak bersatuan; diskalakan agar luasnya pada 0,01-0,5 Hz sama dengan
        # varians IBI (ms^2), sehingga LF dan HF sebanding dengan Welch. LF/HF tidak
        # bergantung pada skala ini.
        pg = lombscargle(t, y, 2 * np.pi * f, precenter=False)
        luas = float(np.trapz(pg, f))
        p = pg * (float(y.var()) / luas) if luas > 0 else np.zeros_like(pg)
    lf, hf = _daya_pita(f, p, PITA_LF), _daya_pita(f, p, PITA_HF)
    return {"lf": lf, "hf": hf, "lf_hf": lf / hf if hf > 0 else float("nan")}


def fitur_jendela(x: np.ndarray, fs: float, s: SetelanHRV) -> dict | None:
    """Fitur untuk satu jendela sinyal pulsa. None bila IBI valid terlalu sedikit."""
    durasi = len(x) / fs
    butuh_frek = any(f in FITUR_FREKUENSI for f in s.fitur)
    if butuh_frek and durasi < s.min_frekuensi_s:
        raise ValueError(
            f"Config meminta {[f for f in s.fitur if f in FITUR_FREKUENSI]} pada jendela "
            f"{durasi:g} s; minimal {s.min_frekuensi_s:g} s."
        )
    t, ibi = ibi_dari_puncak(cari_puncak(filter_pulsa(x, fs), fs), fs)
    n0 = ibi.size
    t, ibi = saring_ibi(t, ibi, s)
    if n0 == 0 or ibi.size < 3 or ibi.size / n0 < s.min_ibi_valid:
        return None
    out = fitur_waktu(ibi)
    if butuh_frek:
        out.update(fitur_frekuensi(t, ibi, durasi, s))
    out["n_ibi"] = int(ibi.size)
    out["proporsi_ibi_valid"] = float(ibi.size / n0)
    return out
