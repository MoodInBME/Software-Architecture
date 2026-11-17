# DISFA dan DISFA+

> Status Pra-TA: sumber label intensitas AU selama akses Aff-Wild2 belum disetujui (ISU-071). Akses lewat formulir perjanjian, belum diajukan per 17 Nov 2026.

| Butir           | DISFA                                                                          | DISFA+                                                                           |
| --------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Pengelola       | Mohammad H. Mahoor, University of Denver                                       | sama                                                                             |
| Akses           | Formulir perjanjian, dikirim pemilik                                           | Formulir terpisah                                                                |
| Isi             | 27 subjek, video stereo 1024 × 768 pada 20 fps saat menonton klip pemicu emosi | 12 AU, ekspresi diperagakan dan spontan                                          |
| Label           | Intensitas 0 sampai 5 untuk AU 1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26        | 12 AU, 5 tingkat                                                                 |
| Landmark bawaan | 66 titik                                                                       | lihat README setelah akses                                                       |
| Sitasi          | S. M. Mavadati dkk., IEEE Trans. Affective Computing 4(2), 2013                | M. Mavadati dkk., CVPR Workshops 2016 (prosiding, hanya sebagai rujukan dataset) |

## Keputusan pemrosesan

- AU7 tidak dilabeli DISFA. Bila daftar AU naskah memuat AU7, mask AU7 selalu 0 dan AU7 tidak dilaporkan, bukan diisi tebakan.
- 66 titik bawaan tidak dipakai. Landmark diekstrak ulang dengan ekstraktor cermin sehingga graf tetap 68 simpul dan domain masukan sama dengan inferensi.
- 20 fps diinterpolasi ke 30 fps sebelum dipotong menjadi jendela 90 frame.
- Pembagian data per subjek (SNxxx), lazimnya 3 lipatan subjek-independen.
- Metrik: ICC(3,1) dan MAE per AU pada skala 0 sampai 5 (`evaluation/affect_metrics.au_report`).
- Parser DISFA+ belum ditulis. Struktur foldernya berbeda dan akan disusun setelah README resmi diterima, bukan dari tebakan.

## Pembanding OpenFace 2.0

OpenFace 2.0 (Baltrušaitis dkk., FG 2018, lisensi akademik nonkomersial) dapat dijalankan luring pada video DISFA sebagai pembanding intensitas AU. OpenFace sendiri dilatih memakai DISFA, sehingga angka pembanding ini tidak setara uji lintas dataset dan harus ditulis demikian. OpenFace tidak dipakai di runtime cermin.
