# Kartu Dataset — UBFC-Phys

## Ringkasan

- **Nama:** UBFC-Phys, basis data multimodal untuk studi psikofisiologis stres sosial [1].
- **Isi:** video wajah 56 partisipan selama tiga tugas (istirahat, berbicara, dan aritmetika), disertai sinyal fisiologis dari gelang di pergelangan tangan (BVP dan EDA).
- **Peran di MoodIn:** dataset utama jalur fisiologis pada fase TA (milestone M1).
- **Status:** tersedia bagi penulis TA.

Struktur berkas, laju _frame_ video, dan laju sampel sinyal **dibaca langsung dari berkas dataset**, bukan diasumsikan. Kode lama di repo ini mengasumsikan 30 fps (UBFC-rPPG); asumsi itu tidak boleh dibawa ke UBFC-Phys tanpa pengecekan.

## Akses dan lisensi

- Akses diperoleh dari pengelola dataset sesuai syarat mereka. Pemakaian dan sitasi mengikuti syarat tersebut.
- **Data mentah maupun turunannya** (video, sinyal, _landmark_, fitur, _cache_) **tidak boleh di-commit** ke repo mana pun di MoodInBME.

## Cara pakai di repo

Pemuat data UBFC-Phys belum ada. Pemuat akan dibuat di `data/loaders/` pada milestone M1, dan bagian ini diperbarui sesudahnya.

## Sitasi

[1] R. Meziati Sabour, Y. Benezeth, P. De Oliveira, J. Chappé, dan F. Yang, "UBFC-Phys: A multimodal database for psychophysiological studies of social stress," _IEEE Transactions on Affective Computing_, vol. 14, no. 1, hal. 622–636, 2023, doi: [10.1109/TAFFC.2021.3056960](https://doi.org/10.1109/TAFFC.2021.3056960).
