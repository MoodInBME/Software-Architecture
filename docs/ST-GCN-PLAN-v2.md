# Rencana Modifikasi ST-GCN v2 (VA-bridge)

Keputusan yang disepakati:
1. **Alur output**: fusi di ruang Valence-Arousal, SATU argmax kuadran di akhir.
2. **Dataset ekspresi**: Aff-Wild2 (utama, in-the-wild + VA) + CK+/Oulu-CASIA (baseline).
3. **Platform testing**: Kaggle (GPU P100/T4 gratis).

## 1. Kenapa alur ini (anti-konflik argmax)

UBFC-rPPG (cabang STC-rPPG) tidak punya label emosi; dataset ekspresi tidak punya
ground-truth PPG. Jadi kedua cabang **tidak bisa dilatih end-to-end bersama**.
Solusi: latih terpisah, satukan di ruang VA.

    video wajah
      |-- ST-GCN   -> (valence, arousal_wajah)
      |-- STC-rPPG -> HR/HRV -> arousal_fisio
    fusion (late):
      valence_final = valence_wajah
      arousal_final = w1*arousal_wajah + w2*arousal_fisio
    -> labels.va_to_mood(valence_final, arousal_final)   # SATU argmax kuadran

rPPG hanya menambah bukti di sumbu arousal (justifikasi fisiologis). Keputusan mood
terjadi sekali -> tidak ada dua argmax yang berkonflik. Bobot w1,w2 dikalibrasi di
dataset ber-video+fisiologi+emosi (MAHNOB-HCI / UBFC-Phys), opsional.

## 2. Realita teknis kunci

- ST-GCN+Transformer = model TEMPORAL -> butuh urutan frame (video). AffectNet
  (gambar statis) hanya untuk pretrain fitur per-frame.
- Harmonisasi label: emosi kategorikal -> titik VA (Russell) via
  `labels.EMOTION_TO_VA`; Aff-Wild2 sudah ber-VA.
- Landmark: MediaPipe 468 -> iBUG 68 (`landmark_map.subset_to_ibug68`), lalu
  normalisasi spasial (pusat hidung, skala antar-mata).

## 3. Kompatibilitas dengan komponen yang ada

Paket ini DISAMAKAN dengan tanda tangan file Anda saat ini:
- `STGCNBlock(in, out, k_num, t_kernel, stride, residual, dropout)`
- `TemporalTransformer(d_model, nhead, num_layers, dim_ff, dropout)`
- `FaceGraph(num_node, edges, strategy, max_hop, center).A -> (K, V, V)`
- `labels.va_to_mood(...) -> str` (API lama dipertahankan; ditambah
  `emotion_to_va`, `va_to_mood_id`, `EMOTION_TO_VA`).

## 4. Rencana bertahap

1. **Sekarang (Kaggle, CK+)**: validasi pipeline end-to-end. CK+ posed, hasil
   hanya untuk memastikan alur benar.
2. **Ajukan EULA Aff-Wild2** (paralel). Setelah disetujui, ganti dataset ke
   Aff-Wild2 (`--layout affwild2`) untuk hasil in-the-wild ber-VA.
3. **Kalibrasi fusi** w1,w2 di MAHNOB-HCI/UBFC-Phys (opsional).

## 5. Catatan validasi

Lingkungan build tidak punya torch/mediapipe -> file hanya divalidasi sintaks
(py_compile). Uji fungsional di Kaggle (jalankan sel `--sanity` dulu).

## 6. CK+ di Kaggle -- peringatan

Banyak mirror CK+ hanya menyimpan frame puncak (gambar tunggal), bukan urutan.
Untuk model temporal, gunakan mirror ber-urutan frame (neutral -> peak).
`extract_landmarks_video.py` menerima folder-frame maupun file video.
