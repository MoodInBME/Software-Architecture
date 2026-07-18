# ST-GCN + Graph-Transformer (Emotion Branch)

Cabang klasifikasi mood berbasis **graf landmark wajah**. Mengubah urutan
landmark (dari MediaPipe FaceMesh) menjadi 5 kelas mood MoodIn:
**Ceria, Cemas, Tertekan, Rileks, Netral**.

## Isi modul

| File | Peran |
|---|---|
| `graph.py` | Definisi graf wajah + adjacency ternormalisasi (uniform/distance/spatial) |
| `st_gcn_block.py` | Blok ST-GCN: graph conv spasial + temporal conv + residual |
| `transformer.py` | Temporal Transformer encoder (bagian "Graph-Tran") |
| `model.py` | Perakitan model `STGCN_GraphTransformer` |
| `labels.py` | Harmonisasi label VA/kategorikal -> 5 kelas mood |
| `train.py` | Kerangka pelatihan + mode `--sanity` |
| `config.yaml` | Hyperparameter |

## Representasi input

Tensor `(N, C, T, V)`:

- `N` batch
- `C` channel per node = 2 (x, y) atau 3 (+ confidence/z)
- `T` jumlah frame per window (default 150 = ~5 dtk @ 30 FPS)
- `V` jumlah node landmark (default 68)

## Uji cepat (butuh PyTorch)

    python train.py --sanity

Menjalankan satu langkah forward + backward dengan data acak untuk memverifikasi
bentuk tensor dan aliran gradien.

## Status

- [x] Graf, blok ST-GCN, transformer, perakitan model, harmonisasi label
- [x] Lulus `py_compile` (validasi sintaks)
- [ ] Belum dieksekusi dengan PyTorch (torch tidak terpasang di lingkungan build)
- [ ] Loader dataset (`data/loaders/dataset_loader.py`) belum dihubungkan
- [ ] Tuning arsitektur graf wajah (edge inter-region) & jumlah node
