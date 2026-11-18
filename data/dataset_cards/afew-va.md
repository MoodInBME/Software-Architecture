# AFEW-VA

> Status Pra-TA: dataset cadangan untuk kepala valensi-arousal selama akses Aff-Wild2 belum disetujui (ISU-071).

| Butir                       | Isi                                                                                                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sumber resmi                | https://ibug.doc.ic.ac.uk/resources/afew-va-database/ (iBUG, Imperial College London)                                                                           |
| Isi                         | 600 klip film, label valensi dan arousal per frame, 68 landmark per frame                                                                                       |
| Lisensi                     | Hanya untuk riset. Tidak ada formulir, tetapi ketentuan tetap mengikat                                                                                          |
| Sitasi wajib                | J. Kossaifi, G. Tzimiropoulos, S. Todorovic, M. Pantic, Image and Vision Computing, 2017; A. Dhall, R. Goecke, S. Lucey, T. Gedeon, IEEE MultiMedia 19(3), 2012 |
| Skala label                 | -10 sampai 10 menurut makalah [Medium, cek min dan maks setelah unduh], dibagi 10 menjadi [-1, 1]                                                               |
| Laju frame                  | Belum dinyatakan di halaman dataset. Tetapkan `--src_fps` setelah diukur                                                                                        |
| Subjek untuk pembagian data | Aktor (kunci `actor`), bukan nomor klip, agar wajah yang sama tidak bocor ke data uji                                                                           |

## Cara pakai

```
python data/loaders/afew_va.py download --dest <di luar repo> --saya-setuju-riset
python data/loaders/afew_va.py prepare --root <folder ekstrak> --output_dir <di luar repo> --landmarks extract
```

`--landmarks extract` (bawaan) memakai ekstraktor yang sama dengan cermin. Landmark anotasi (`gt`) dipakai untuk mengevaluasi ekstraktor.

## Keterbatasan

- Klip film, bukan wajah di depan cermin: pose, pencahayaan, dan ekspresi lebih bervariasi daripada kondisi penelitian.
- Klip pendek; klip yang lebih pendek dari jendela diperpanjang oleh dataset loader.
- Tidak ada label AU, sehingga klip AFEW-VA hanya dipakai untuk keluaran valensi-arousal.
- Data dan klip turunan tidak boleh masuk repo, Kaggle publik, atau dibagikan ulang.
