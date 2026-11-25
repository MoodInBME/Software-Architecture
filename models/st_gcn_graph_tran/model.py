"""
ST-GCN + Graph-Transformer untuk cabang ekspresi MoodIn.

Revisi (VA-bridge) atas versi awal:
  * Head fleksibel via `head_type` (va | class | dual). Default 'va' (regresi
    valence-arousal) sesuai alur fusion yang disepakati.
  * forward() mengembalikan dict: {'va': (B,2)} atau {'logits': (B,5)} atau keduanya.
  * head_type 'au_va': {'au': (B, n_au), 'va': (B, 2)}; n_au = panjang au_list di config.
  * Susunan unit diatur ``channels`` dan ``strides``; ``use_transformer=False`` mengganti
    Transformer temporal dengan rerata waktu bermask. Rancangan Pra-TA (B.3): empat unit
    64-64-128-128, stride 2 di unit ke-3, tanpa Transformer. Nilai bawaan konstruktor tetap
    susunan lama (enam unit + Transformer) agar bobot yang sudah ada masih bisa dimuat;
    config.yaml memakai rancangan B.3.

Kompatibel dengan komponen yang sudah ada:
  * st_gcn_block.STGCNBlock(in, out, k_num, t_kernel, stride, residual, dropout)
  * transformer.TemporalTransformer(d_model, nhead, num_layers, dim_ff, dropout)
  * graph.FaceGraph(num_node, edges, strategy, max_hop, center).A -> (K, V, V)

Butuh PyTorch (tidak dieksekusi di sandbox; lulus py_compile untuk sintaks).
"""
import torch
import torch.nn as nn

try:
    from .graph import FaceGraph
    from .st_gcn_block import STGCNBlock
    from .transformer import TemporalTransformer
    from .heads import build_head
except ImportError:  # dijalankan sebagai skrip lepas
    from graph import FaceGraph
    from st_gcn_block import STGCNBlock
    from transformer import TemporalTransformer
    from heads import build_head


CHANNELS_LAMA = (64, 64, 128, 128, 256, 256)
STRIDES_LAMA = (1, 1, 2, 1, 2, 1)
CHANNELS_B3 = (64, 64, 128, 128)
STRIDES_B3 = (1, 1, 2, 1)


class STGCN_GraphTransformer(nn.Module):
    def __init__(self, in_channels=2, num_class=5, num_node=68,
                 graph_strategy="spatial", max_hop=1, t_kernel=9,
                 transformer_layers=2, transformer_heads=8, transformer_ff=512,
                 dropout=0.3, head_type="va", n_au=12,
                 channels=CHANNELS_LAMA, strides=STRIDES_LAMA,
                 use_transformer=True, block_dropout=0.0):
        super().__init__()
        channels, strides = tuple(int(c) for c in channels), tuple(int(s) for s in strides)
        if not channels or len(channels) != len(strides):
            raise ValueError(
                f"channels {channels} dan strides {strides} harus sama panjang dan tidak kosong"
            )
        self.channels, self.strides = channels, strides
        self.use_transformer = bool(use_transformer)
        self.head_type = head_type.lower()
        graph = FaceGraph(num_node=num_node, strategy=graph_strategy, max_hop=max_hop)
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.register_buffer("A", A)
        k_num = A.size(0)

        self.data_bn = nn.BatchNorm1d(in_channels * num_node)
        blok, c_in = [], in_channels
        for i, (c_out, s) in enumerate(zip(channels, strides)):
            # Unit pertama tanpa residual, mengikuti Yan dkk.: kanal masukan (x, y) terlalu
            # sedikit untuk dijumlahkan langsung dengan 64 kanal keluaran.
            blok.append(STGCNBlock(c_in, c_out, k_num, t_kernel, stride=s,
                                   residual=i > 0, dropout=block_dropout))
            c_in = c_out
        self.blocks = nn.ModuleList(blok)
        d = channels[-1]
        self.temporal_transformer = None
        if self.use_transformer:
            self.temporal_transformer = TemporalTransformer(
                d_model=d, nhead=transformer_heads,
                num_layers=transformer_layers, dim_ff=transformer_ff, dropout=dropout,
            )
        self.head = build_head(self.head_type, d, num_class=num_class, dropout=dropout,
                               n_au=n_au)

    def temporal_strides(self):
        return [int(getattr(blk, "stride", 1)) for blk in self.blocks]

    def downsample_mask(self, frame_mask: torch.Tensor) -> torch.Tensor:
        """Mask frame (N, T) -> mask posisi waktu setelah blok ber-stride (N, T').

        Konvolusi temporal ber-stride s dengan padding (k - 1) / 2 menghasilkan
        ceil(T / s) posisi yang berpusat di frame 0, s, 2s, ...; mask diambil di
        frame pusat yang sama.
        """
        m = frame_mask.bool()
        for s in self.temporal_strides():
            if s > 1:
                m = m[:, ::s]
        return m

    def pool_temporal(self, feats: torch.Tensor, mask=None) -> torch.Tensor:
        """(Transformer temporal bila aktif) + rerata waktu. feats (N, T', C); mask (N, T') bool.

        Dengan mask, posisi padding tidak diperhatikan Transformer dan tidak ikut dirata-rata.
        """
        tr = self.temporal_transformer
        if mask is None:
            return (tr(feats) if tr is not None else feats).mean(dim=1)
        if mask.shape != feats.shape[:2]:
            raise ValueError(f"mask {tuple(mask.shape)} tidak cocok dengan fitur {tuple(feats.shape[:2])}")
        mask = mask.clone()
        mask[:, 0] |= ~mask.any(dim=1)  # jaga agar tiap sampel punya minimal satu posisi
        h = tr(feats, key_padding_mask=~mask) if tr is not None else feats
        w = mask.to(h.dtype).unsqueeze(-1)
        return (h * w).sum(dim=1) / w.sum(dim=1)

    def forward(self, x, frame_mask=None):
        """x: (N, C, T, V). frame_mask: (N, T), 1 = frame asli, 0 = padding (opsional)."""
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)
        for blk in self.blocks:
            x = blk(x, self.A)
        x = x.mean(dim=3)                      # (N, C, T')  rata-rata spasial
        x = x.permute(0, 2, 1).contiguous()    # (N, T', C)
        mask = None if frame_mask is None else self.downsample_mask(frame_mask)
        x = self.pool_temporal(x, mask)        # (N, C) global temporal pooling
        out = self.head(x)
        if isinstance(out, dict):
            return out
        if self.head_type == "va":
            return {"va": out}
        return {"logits": out}


if __name__ == "__main__":
    for ht in ("va", "class", "dual", "au_va"):
        m = STGCN_GraphTransformer(head_type=ht)
        y = m(torch.randn(2, 2, 150, 68))
        print(ht, {k: tuple(v.shape) for k, v in y.items()})
