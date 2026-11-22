# `interaction/` — gestur tangan

Kursor dan pemilihan di layar cermin tanpa sentuhan. Status: **usulan** (ADR-16 di MoodIn-master berstatus Diusulkan).

## Alur

1. `hand_tracker.PelacakTangan`: MediaPipe Hands 0.10.14 (`mp.solutions.hands`, satu tangan, `model_complexity=0`) memberi 21 titik sendi per frame. Frame hanya diproses di memori.
2. `gesture_features`: titik digeser relatif ke pergelangan dan dibagi ukuran telapak; rasio cubit (ujung ibu jari ke ujung telunjuk); jari lurus; telapak terbuka.
3. `one_euro.OneEuro2D`: peredam getar kursor (Casiez dkk., CHI 2012).
4. `gesture_state.MesinGestur`: keadaan diam, menunjuk, memilih. Cubit dengan histeresis dua ambang mengirim `select` sekali; telapak terbuka ditahan 1 detik mengirim `back`; tangan hilang mengirim `hand_lost`; selama fase ukur tidak ada peristiwa. Kursor diambil dari pangkal telunjuk (titik 5) agar tidak bergeser saat mencubit.
5. `gesture_state.PenomorPeristiwa`: peristiwa menjadi pesan `moodin/gesture/event` kontrak IPC 2.0.0 dengan `seq` naik. Titik sendi tidak pernah dikirim.

Ambang bawaan (`SetelanGestur`) adalah titik awal dan perlu dikalibrasi dengan `scripts/demo_gestur.py`, yang menampilkan rasio cubit di layar.

## Demo dan uji

```powershell
.\.venv-gpu\Scripts\python.exe scripts\demo_gestur.py
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_gestur.py -q
```

Uji memakai deret titik sendi sintetis, tanpa video.

## Inspirasi

Implementasi ini ditulis sendiri. Gagasan fitur tangan ternormalisasi dan kursor virtual dengan cubit terinspirasi dari dua repositori berlisensi Apache-2.0: [kinivi/hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe) dan [Kazuhito00/simple-virtual-mouse-using-mediapipe](https://github.com/Kazuhito00/simple-virtual-mouse-using-mediapipe). Tidak ada kode yang disalin dari keduanya, dan modul ini tidak menggerakkan tetikus sistem operasi.
