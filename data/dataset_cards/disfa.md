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

## Catatan pemrosesan

- AU yang tidak dilabeli DISFA tidak diisi tebakan; label itu ditandai tidak tersedia.
- Landmark diekstrak ulang dengan ekstraktor cermin, bukan memakai titik bawaan dataset.
- Laju frame disamakan dengan laju frame cermin sebelum dipotong menjadi jendela.
- Pembagian data per subjek (SNxxx).
- Parser DISFA+ belum ditulis dan akan disusun setelah README resmi diterima.

## Pembanding OpenFace 2.0

OpenFace 2.0 (Baltrušaitis dkk., FG 2018, lisensi akademik nonkomersial) dapat dijalankan luring pada video DISFA sebagai pembanding intensitas AU. OpenFace sendiri dilatih memakai DISFA, sehingga angka pembanding ini tidak setara uji lintas dataset dan harus ditulis demikian. OpenFace tidak dipakai di runtime cermin.
