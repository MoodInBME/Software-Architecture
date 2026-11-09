"""Konstanta bersama jalur fisiologis (STC-rPPG).

``HR_BAND_HZ`` adalah pita frekuensi denyut jantung yang dipakai di seluruh
jalur fisiologis: filter Butterworth pascaproses, rentang spektrum pada rugi
domain frekuensi, dan estimasi BPM saat inferensi. Nilainya 0,75-4,0 Hz
(45-240 bpm), sama dengan pita yang dipakai Lin dkk. (2025, IEEE TIM, bagian
IV-B) untuk STC-rPPG.

Nilai ini sengaja hanya ditulis di sini. Modul lain mengimpornya, dan
``config.yaml`` tidak memuat salinannya, supaya pita filter dan pita rugi
tidak bisa berbeda tanpa disadari.
"""

HR_BAND_HZ: tuple[float, float] = (0.75, 4.0)
"""Batas bawah dan atas pita denyut jantung dalam Hz."""
