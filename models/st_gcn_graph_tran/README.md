# ST-GCN + Graph-Transformer (Emotion Branch)

Cabang ekspresi MoodIn berbasis **graf landmark wajah**. Mengubah urutan landmark
(dari MediaPipe FaceMesh) menjadi **regresi Valence–Arousal**, lalu dipetakan ke
5 mood MoodIn (**Ceria, Cemas, Tertekan, Rileks, Netral**) via satu argmax kuadran.

> **Alur VA-bridge:** cabang ini mengeluarkan `(valence, arousal)`, **bukan** langsung
> 5 kelas. Klasifikasi mood terjadi sekali di modul fusi (`labels.va_to_mood`) setelah
> arousal digabung dengan sinyal fisiologis STC-rPPG. Ini mencegah konflik argmax saat fusion.

## Isi modul

| File                | Peran                                                                                                 |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| `graph.py`        | Definisi graf wajah + adjacency ternormalisasi (uniform/distance/spatial)                             |
| `st_gcn_block.py` | Blok ST-GCN: graph conv spasial + temporal conv + residual                                            |
| `transformer.py`  | Temporal Transformer encoder (bagian "Graph-Tran")                                                    |
| `model.py`        | Perakitan`STGCN_GraphTransformer` — `head_type` (va/class/dual), `forward()` → **dict** |
| `heads.py`        | Kepala keluaran:`VAHead` (→ tanh 2), `ClassHead` (→ 5 logits), `DualHead`                     |
| `losses.py`       | `CCCLoss` (Concordance CC, standar Aff-Wild2/ABAW) + `DualLoss`                                   |
| `labels.py`       | Harmonisasi label +`EMOTION_TO_VA`, `va_to_mood()` (→str), `va_to_mood_id()` (→int)           |
| `train.py`        | Trainer penuh: regresi VA, loss CCC, evaluasi, checkpoint, mode`--sanity`                           |
| `config.yaml`     | Hyperparameter (termasuk`head_type`, `transformer_ff`, `neutral_radius`, blok `loss`)         |

> `graph.py`, `st_gcn_block.py`, `transformer.py` = versi tim (tidak diubah pada push ini).

## Tanda tangan komponen (acuan integrasi)

```
FaceGraph(num_node=68, edges=None, strategy="spatial", max_hop=1, center=33).A  # (K, V, V)
STGCNBlock(in_channels, out_channels, k_num, t_kernel=9, stride=1, residual=True, dropout=0.0)
TemporalTransformer(d_model, nhead=8, num_layers=2, dim_ff=512, dropout=0.1)
```

## Representasi input

Tensor `(N, C, T, V)`:

- `N` batch
- `C` channel per node = 2 (x, y) atau 3 (+ confidence/z)
- `T` jumlah frame per window (default 150 = ~5 dtk @ 30 FPS)
- `V` jumlah node landmark (default 68 — MediaPipe 468 → `subset_to_ibug68`)

## Keluaran model (BREAKING)

`forward()` mengembalikan **dict**, bukan tensor:

- `head_type="va"`    → `{"va": (N, 2)}`  (valence, arousal ∈ [-1, 1])
- `head_type="class"` → `{"logits": (N, 5)}`
- `head_type="dual"`  → `{"va": (N, 2), "logits": (N, 5)}`

Kode lama `model(x).argmax(1)` harus diganti menjadi `model(x)["va"]` / `model(x)["logits"]`.

## Uji cepat (butuh PyTorch)

    python -m models.st_gcn_graph_tran.train --sanity

Menjalankan satu langkah forward + backward dengan data acak untuk memverifikasi
bentuk tensor dan aliran gradien (loss CCC untuk head VA).

## Latih (baseline CK+)

    # 1) ekstraksi + cache landmark (SEKALI)
    python data/preprocessing/extract_landmarks_video.py
    --input_root dataset/ckplus --layout ckplus
    --output_dir data/prepared/ckplus --clip_len 150

    # 2) latih regresi VA
    python -m models.st_gcn_graph_tran.train
    --config models/st_gcn_graph_tran/config.yaml
    --manifest data/prepared/ckplus/manifest.json --val_split 0.2

Untuk **Aff-Wild2** (setelah EULA): `--layout affwild2`. Testing GPU gratis: lihat
`notebooks/ck_plus_stgcn_kaggle.ipynb`. Metrik evaluasi (CCC valence/arousal, akurasi
kuadran mood, macro-F1) di `evaluation/va_metrics.py`.

## Status

- [X] Graf, blok ST-GCN, transformer, perakitan model, harmonisasi label
- [X] Head VA + loss CCC + trainer penuh (VA-bridge)
- [X] Lulus `py_compile` (validasi sintaks)
- [ ] Belum dieksekusi dengan PyTorch (torch tidak terpasang di lingkungan build) — uji di Kaggle
- [ ] Verifikasi visual peta landmark 468 → 68 sebelum training sungguhan
- [ ] Selaraskan `CATEGORICAL_MAP` vs koordinat `EMOTION_TO_VA` untuk emosi ambigu
- [ ] Tuning arsitektur graf wajah (edge inter-region) & jumlah node
