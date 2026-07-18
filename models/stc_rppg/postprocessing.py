"""Post-processing rPPG (versi diperbaiki - lihat S3/S9 di CODE_REVIEW_root_files.md).

Perubahan dari versi asli:
- Filter Butterworth via second-order sections (sosfiltfilt) -> stabil numerik.
- Estimasi BPM dibatasi ke pita HR, ber-zero-pad, ber-jendela Hanning, plus
  interpolasi parabola sub-bin -> resolusi jauh lebih halus dari 14 BPM/bin.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt


class rPPGProcessor:
    def __init__(self, fps=30.0, lowcut=0.7, highcut=3.5, order=4):
        self.fps = float(fps)
        self.lowcut = float(lowcut)
        self.highcut = float(highcut)
        self.order = int(order)
        self._sos = self._design_filter()

    def _design_filter(self):
        nyq = 0.5 * self.fps
        low = self.lowcut / nyq
        high = self.highcut / nyq
        return butter(self.order, [low, high], btype="band", output="sos")

    def filter_signal(self, data):
        """Bandpass zero-phase. Membutuhkan sinyal cukup panjang (>= ~ 3*order*2)."""
        x = np.asarray(data, dtype=np.float64)
        x = x - np.mean(x)
        return sosfiltfilt(self._sos, x)

    def calculate_bpm(self, filtered_signal, n_fft=4096):
        x = np.asarray(filtered_signal, dtype=np.float64)
        if x.size < 4:
            return 0.0
        x = x - x.mean()
        x = x * np.hanning(x.size)
        n_fft = int(max(n_fft, x.size))
        mag = np.abs(np.fft.rfft(x, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.fps)

        band = (freqs >= self.lowcut) & (freqs <= self.highcut)
        if not np.any(band):
            return 0.0
        band_idx = np.where(band)[0]
        peak = band_idx[int(np.argmax(mag[band]))]
        f_peak = self._parabolic_interp(mag, peak, freqs)
        return float(f_peak * 60.0)

    @staticmethod
    def _parabolic_interp(mag, k, freqs):
        if k <= 0 or k >= len(mag) - 1:
            return freqs[k]
        a, b, c = mag[k - 1], mag[k], mag[k + 1]
        denom = (a - 2.0 * b + c)
        if denom == 0:
            return freqs[k]
        delta = 0.5 * (a - c) / denom
        return freqs[k] + delta * (freqs[1] - freqs[0])


if __name__ == "__main__":
    T, fps = 300, 30.0
    t = np.linspace(0, T / fps, T, endpoint=False)
    hr = 1.5  # 90 BPM
    raw = np.sin(2 * np.pi * hr * t) + 0.5 * np.random.randn(T)
    proc = rPPGProcessor(fps=fps)
    clean = proc.filter_signal(raw)
    print(f"Target : {hr * 60:.1f} BPM")
    print(f"Estimasi: {proc.calculate_bpm(clean):.2f} BPM")
