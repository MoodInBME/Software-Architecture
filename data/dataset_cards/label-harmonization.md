# Harmonisasi Label ke Lima Kategori Mood

> Status tahap Pra-TA: ringkasan tingkat tinggi. Aturan pemetaan yang berlaku ada di naskah Proposal Tugas Akhir (Pra-TA) dan di `models/st_gcn_graph_tran/mood_rules.py`. Versi era GEMASTIK dapat dilihat pada tag `gemastik-xix-2026`.

## Kategori

Lima kategori mood: Ceria, Rileks, Netral, Cemas, dan Tertekan. Jalur wajah mengeluarkan valensi dan arousal kontinu dalam rentang [-1, 1]; kategori ditentukan di hilir oleh aturan rancangan Pra-TA (`data.mood_rule: tabel33`). Aturan lama berbentuk lingkaran tetap tersedia sebagai `lingkaran` untuk arsip dan perbandingan.

Kombinasi label yang belum tercakup aturan tidak dipetakan, dikeluarkan dari metrik kategori, dan jumlahnya dilaporkan. Perlakuan akhirnya ditetapkan bersama dosen pembimbing.

## Peran dataset

| Dataset         | Ruang label asli                    | Dipakai untuk                        |
| --------------- | ----------------------------------- | ------------------------------------ |
| UBFC-rPPG, PURE | Sinyal denyut (tanpa label emosi)   | Jalur fisiologis saja                |
| UBFC-Phys       | Sinyal fisiologis dan kondisi tugas | Jalur stres                          |
| AFEW-VA         | Valensi dan arousal kontinu         | Keluaran valensi-arousal jalur wajah |
| DISFA           | Intensitas Action Unit              | Keluaran AU jalur wajah              |

## Catatan

- Data sinyal denyut tidak membawa label mood; ruang label jalur fisiologis dan jalur wajah tidak dicampur saat menghitung rugi.
- Skala valensi-arousal setiap dataset dinormalisasi ke [-1, 1] sebelum pemetaan kategori.
- Skor `decision_margin` di runtime adalah jarak ke batas kategori, bukan peluang, dan tidak ditampilkan sebagai persen.
