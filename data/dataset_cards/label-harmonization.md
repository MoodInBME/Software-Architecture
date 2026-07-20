# Harmonisasi Label -> 5 Kelas Mood (Kuadran Valence-Arousal)

Dataset MoodIn punya ruang label heterogen. Dokumen ini menyatukan semuanya ke skema tunggal
(kelemahan W2). **Ini asumsi kerja - perlu disepakati tim & diselaraskan dgn proposal.**

## Target akhir: 5 kelas

| Kelas | Kuadran V-A | Valence | Arousal |
| --- | --- | --- | --- |
| Ceria | Q1 | + | + |
| Cemas | Q2 | - | + |
| Tertekan | Q3 | - | - |
| Rileks | Q4 | + | - |
| Netral | pusat | ~0 | ~0 |

Aturan pusat: jika |valence| < `r` **dan** |arousal| < `r` (mis. `r=0.2` pada skala [-1,1]) -> **Netral**.

## Peran tiap dataset `[Sedang]`

| Dataset | Ruang label asli | Dipakai untuk | Pemetaan |
| --- | --- | --- | --- |
| UBFC-rPPG | PPG/HR (tanpa emosi) | Pra-latih STC-rPPG | - (bukan label mood) |
| PURE | PPG/HR (tanpa emosi) | Validasi rPPG | - |
| AffectNet | Kategorikal + V-A kontinu | ST-GCN / fusi | Pakai V-A kontinu -> kuadran |
| CK+ | 7 emosi kategorikal | ST-GCN (ekspresi) | Peta kategorikal -> kuadran (lihat bawah) |

## Peta kategorikal -> kuadran (contoh, verifikasi) `[Sedang]`

| Emosi | Kuadran | Kelas MoodIn |
| --- | --- | --- |
| happy | Q1 | Ceria |
| fear / anger | Q2 | Cemas |
| sad / disgust | Q3 | Tertekan |
| calm / content | Q4 | Rileks |
| neutral | pusat | Netral |
| surprise | ambigu (Q1/Q2) | ⚠️ putuskan aturan (default: Q2 bila arousal tinggi + valence rendah) |

## Catatan penting

- **rPPG != emosi.** UBFC/PURE hanya melatih cabang fisiologis; label mood datang dari dataset ekspresi + V-A. Jangan campur ruang label saat menghitung loss klasifikasi. `[Tinggi]`
- Skala V-A antar dataset berbeda (mis. [-1,1] vs [1,9]) -> **normalisasi ke [-1,1] dulu** sebelum kuadran. `[Tinggi]`
- 'surprise' & batas Netral (`r`) adalah dua keputusan yang paling memengaruhi confusion matrix - kunci nilainya di `config/default.yaml`.
