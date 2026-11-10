"""Uji pita denyut jantung bersama (HR_BAND_HZ) di jalur fisiologis.

Pita 0,75-4,0 Hz harus dipakai oleh filter pascaproses, rugi domain frekuensi,
estimasi BPM saat inferensi, dan skrip pelatihan, tanpa salinan nilai di
tempat lain.

Jalankan dari root repo dengan Python 3.12:

    python -m unittest tests.test_hr_band -v
"""

import importlib.util
import inspect
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from models.stc_rppg import loss as loss_mod
from models.stc_rppg import train as train_mod
from models.stc_rppg.constants import HR_BAND_HZ
from models.stc_rppg.loss import STCHybridLoss
from models.stc_rppg.postprocessing import rPPGProcessor

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0
DURASI_S = 60.0
TEPI = 300  # 10 s di tiap ujung dibuang agar efek tepi sosfiltfilt tidak ikut terukur
SEED = 42

ADA_CV2 = importlib.util.find_spec("cv2") is not None


def _waktu():
    return np.arange(int(DURASI_S * FPS)) / FPS


def _gain(proc, f_hz):
    """Rasio amplitudo keluaran/masukan filter untuk sinus murni f_hz."""
    y = proc.filter_signal(np.sin(2 * np.pi * f_hz * _waktu()))
    return float(np.sqrt(2.0) * y[TEPI:-TEPI].std())


class TestKonstanta(unittest.TestCase):
    def test_nilai_pita(self):
        self.assertEqual(HR_BAND_HZ, (0.75, 4.0))

    def test_bawaan_modul_mengacu_konstanta(self):
        sig = inspect.signature(rPPGProcessor.__init__).parameters
        self.assertEqual((sig["lowcut"].default, sig["highcut"].default), HR_BAND_HZ)
        sig = inspect.signature(STCHybridLoss.__init__).parameters
        self.assertEqual((sig["low"].default, sig["high"].default), HR_BAND_HZ)


class TestFilterBandpass(unittest.TestCase):
    def setUp(self):
        self.proc = rPPGProcessor(fps=FPS)

    def test_meloloskan_1_2_hz(self):
        self.assertGreater(_gain(self.proc, 1.2), 0.98)

    def test_meredam_0_5_hz(self):
        self.assertLess(_gain(self.proc, 0.5), 0.05)

    def test_meredam_5_hz(self):
        self.assertLess(_gain(self.proc, 5.0), 0.10)

    def test_gain_di_frekuensi_potong(self):
        # Butterworth -3 dB dilewatkan dua kali (zero-phase), jadi gain 0,5 di batas pita.
        for f in HR_BAND_HZ:
            with self.subTest(f=f):
                self.assertAlmostEqual(_gain(self.proc, f), 0.5, delta=0.02)

    def test_pita_tidak_valid_ditolak(self):
        for low, high, fps in [(4.0, 0.75, 30.0), (0.0, 4.0, 30.0), (0.75, 16.0, 30.0)]:
            with self.subTest(low=low, high=high, fps=fps):
                with self.assertRaises(ValueError):
                    rPPGProcessor(fps=fps, lowcut=low, highcut=high)


class TestEstimasiBPM(unittest.TestCase):
    def setUp(self):
        self.proc = rPPGProcessor(fps=FPS)
        self.rng = np.random.default_rng(SEED)

    def _bpm(self, f_hz, noise=0.3):
        raw = np.sin(2 * np.pi * f_hz * _waktu())
        raw = raw + noise * self.rng.standard_normal(raw.size)
        return self.proc.calculate_bpm(self.proc.filter_signal(raw))

    def test_60_bpm(self):
        self.assertAlmostEqual(self._bpm(1.0), 60.0, delta=0.5)

    def test_228_bpm_di_luar_pita_lama(self):
        # 3,8 Hz berada di atas batas lama 3,5 Hz; pita baru harus menangkapnya.
        self.assertAlmostEqual(self._bpm(3.8), 228.0, delta=0.5)

    def test_puncak_di_luar_pita_diabaikan(self):
        # Komponen kuat 0,5 Hz (30 bpm) di bawah pita tidak boleh terpilih.
        t = _waktu()
        raw = 3.0 * np.sin(2 * np.pi * 0.5 * t) + np.sin(2 * np.pi * 1.5 * t)
        bpm = self.proc.calculate_bpm(raw)
        self.assertAlmostEqual(bpm, 90.0, delta=0.5)

    def test_estimate_bpm_sama_dengan_filter_lalu_hitung(self):
        raw = np.sin(2 * np.pi * 1.3 * _waktu()) + 0.3 * self.rng.standard_normal(int(DURASI_S * FPS))
        harap = self.proc.calculate_bpm(self.proc.filter_signal(raw))
        self.assertEqual(self.proc.estimate_bpm(raw), harap)

    def test_klip_128_frame_di_antara_bin(self):
        # 128 frame pada 30 fps: jarak bin FFT tanpa zero-padding 14,1 bpm.
        # 77,4 bpm jatuh di antara bin 70,3 dan 84,4 bpm; estimator harus tetap dekat.
        t = np.arange(128) / FPS
        bpm = self.proc.estimate_bpm(np.sin(2 * np.pi * (77.4 / 60.0) * t))
        self.assertAlmostEqual(bpm, 77.4, delta=1.0)

    def test_sinyal_tidak_layak_menghasilkan_nol(self):
        t = np.arange(128) / FPS
        nan_sig = np.sin(2 * np.pi * 1.2 * t)
        nan_sig[10] = np.nan
        kasus = {
            "terlalu pendek": np.sin(2 * np.pi * 1.2 * t[: self.proc.min_length - 1]),
            "memuat NaN": nan_sig,
            "datar": np.full(128, 0.5),
        }
        for nama, sig in kasus.items():
            with self.subTest(kasus=nama):
                self.assertEqual(self.proc.estimate_bpm(sig), 0.0)


class TestRugiFrekuensi(unittest.TestCase):
    def test_bin_pita_rugi(self):
        n = 300  # 10 s pada 30 fps, resolusi 0,1 Hz
        crit = STCHybridLoss(fps=FPS)
        logpsd = crit._band_logpsd(torch.zeros(1, n))
        freqs = np.fft.rfftfreq(n, d=1.0 / FPS)
        harap = int(np.sum((freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])))
        self.assertEqual(logpsd.shape, (1, harap))
        self.assertEqual(harap, 33)  # 0,8 sampai 4,0 Hz

    def test_rugi_nol_untuk_sinyal_identik(self):
        torch.manual_seed(SEED)
        t = torch.arange(300) / FPS
        sig = torch.sin(2 * torch.pi * 3.8 * t).unsqueeze(0)
        sig = sig + 0.05 * torch.randn_like(sig)
        total, l_time, l_freq = STCHybridLoss(fps=FPS)(sig, sig.clone())
        self.assertLess(float(l_time), 1e-5)
        self.assertLess(float(l_freq), 1e-5)

    def test_modul_rugi_mengimpor_konstanta(self):
        self.assertIs(loss_mod.HR_BAND_HZ, HR_BAND_HZ)


class TestConfigPelatihan(unittest.TestCase):
    def test_config_repo_tanpa_kunci_pita(self):
        cfg = train_mod.load_config(REPO / "models" / "stc_rppg" / "config.yaml")
        for sec, key in train_mod._LEGACY_BAND_KEYS:
            self.assertNotIn(key, cfg.get(sec) or {})

    def test_kunci_lama_ditolak(self):
        for sec, key in train_mod._LEGACY_BAND_KEYS:
            with self.subTest(kunci=f"{sec}.{key}"):
                cfg = {"loss": {"fps": 30.0}, "postprocessing": {"fps": 30.0}}
                cfg[sec][key] = 0.7
                with self.assertRaises(ValueError):
                    train_mod.check_no_legacy_band_keys(cfg)

    def test_bagian_kosong_diterima(self):
        train_mod.check_no_legacy_band_keys(yaml.safe_load("loss:\npostprocessing:\n"))


@unittest.skipUnless(ADA_CV2, "OpenCV tidak terpasang; inference.py butuh cv2")
class TestInferensi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import inference

        cls.inf = inference

    def test_bawaan_mengacu_konstanta(self):
        sig = inspect.signature(self.inf.bpm_from_signal).parameters
        self.assertEqual((sig["f_lo"].default, sig["f_hi"].default), HR_BAND_HZ)

    def test_228_bpm(self):
        sig = np.sin(2 * np.pi * 3.8 * _waktu())
        # Lewat rPPGProcessor.estimate_bpm(): zero-padding 4096 titik dan
        # interpolasi parabola, jadi galat jauh di bawah 1 bpm.
        self.assertAlmostEqual(self.inf.bpm_from_signal(sig, fps=FPS), 228.0, delta=0.5)

    def test_satu_jalur_dengan_processor(self):
        # ISU-062: inferensi dan evaluasi train.py memakai estimator yang sama.
        rng = np.random.default_rng(SEED)
        t = np.arange(128) / FPS
        sig = np.sin(2 * np.pi * 1.29 * t) + 0.2 * rng.standard_normal(t.size)
        harap = rPPGProcessor(fps=FPS).estimate_bpm(sig)
        self.assertEqual(self.inf.bpm_from_signal(sig, fps=FPS), harap)

    def test_klip_128_frame_tidak_terkuantisasi(self):
        # Versi lama memberi 84,4 bpm (bin terdekat) untuk sinus 77,4 bpm.
        t = np.arange(128) / FPS
        sig = np.sin(2 * np.pi * (77.4 / 60.0) * t)
        self.assertAlmostEqual(self.inf.bpm_from_signal(sig, fps=FPS), 77.4, delta=1.0)


if __name__ == "__main__":
    unittest.main()
