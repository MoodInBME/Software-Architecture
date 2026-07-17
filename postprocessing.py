import numpy as np
from scipy.signal import butter, filtfilt

class rPPGProcessor:
    """
    Modul untuk membersihkan sinyal rPPG mentah dan menghitung estimasi Detak Jantung (BPM).
    """
    def __init__(self, fps=30.0, lowcut=0.75, highcut=4.0, order=5):
        self.fps = fps
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def _butter_bandpass(self):
        """
        Menghitung koefisien numerator (b) dan denominator (a) untuk filter Butterworth.
        """
        nyq = 0.5 * self.fps
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return b, a

    def filter_signal(self, data):
        """
        Menerapkan zero-phase Butterworth bandpass filter pada sinyal 1D.
        """
        b, a = self._butter_bandpass()
        # Menggunakan filtfilt agar tidak terjadi pergeseran fasa pada sinyal fisiologis
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def calculate_bpm(self, filtered_signal):
        """
        Menghitung nilai Heart Rate (BPM) dari sinyal yang sudah difilter menggunakan FFT.
        """
        n = len(filtered_signal)
        
        # Melakukan Transformasi Fourier
        fft_data = np.fft.rfft(filtered_signal)
        fft_mag = np.abs(fft_data)
        
        # Mendapatkan array frekuensi yang bersesuaian dengan hasil FFT
        freqs = np.fft.rfftfreq(n, d=1.0/self.fps)
        
        # Mencari indeks frekuensi dengan energi tertinggi
        dominant_freq_idx = np.argmax(fft_mag)
        dominant_freq = freqs[dominant_freq_idx]
        
        # Mengonversi frekuensi (Hz) menjadi Detak per Menit (BPM)
        bpm = dominant_freq * 60.0
        return bpm

# ==========================================
# PENGUJIAN POST-PROCESSING
# ==========================================
if __name__ == "__main__":
    print("=== Menguji Modul Post-Processing ===")
    
    # 1. Mensimulasikan sinyal rPPG mentah dari output model
    # (Misal: Sinyal detak jantung ideal 1.5 Hz atau 90 BPM, ditambah noise acak)
    T = 128
    fps = 30.0
    t = np.linspace(0, T/fps, T, endpoint=False)
    
    ideal_hr_freq = 1.5 # 90 BPM
    raw_signal = np.sin(2 * np.pi * ideal_hr_freq * t) + 0.5 * np.random.randn(T)
    
    # 2. Inisialisasi prosesor
    processor = rPPGProcessor(fps=fps)
    
    # 3. Filter sinyal
    clean_signal = processor.filter_signal(raw_signal)
    
    # 4. Hitung BPM
    estimated_bpm = processor.calculate_bpm(clean_signal)
    
    print(f"Target Detak Jantung Asli : {ideal_hr_freq * 60} BPM")
    print(f"Estimasi Model via FFT    : {estimated_bpm:.2f} BPM")