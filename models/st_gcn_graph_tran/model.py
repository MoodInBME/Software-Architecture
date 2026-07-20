"""
ST-GCN + Graph-Transformer untuk cabang ekspresi MoodIn.

Revisi (VA-bridge) atas versi awal:
  * Head fleksibel via `head_type` (va | class | dual). Default 'va' (regresi
    valence-arousal) sesuai alur fusion yang disepakati.
  * forward() mengembalikan dict: {'va': (B,2)} atau {'logits': (B,5)} atau keduanya.

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


class STGCN_GraphTransformer(nn.Module):
    def __init__(self, in_channels=2, num_class=5, num_node=68,
                 graph_strategy="spatial", max_hop=1, t_kernel=9,
                 transformer_layers=2, transformer_heads=8, transformer_ff=512,
                 dropout=0.3, head_type="va"):
        super().__init__()
        self.head_type = head_type.lower()
        graph = FaceGraph(num_node=num_node, strategy=graph_strategy, max_hop=max_hop)
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.register_buffer("A", A)
        k_num = A.size(0)

        self.data_bn = nn.BatchNorm1d(in_channels * num_node)
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, k_num, t_kernel, residual=False),
            STGCNBlock(64, 64, k_num, t_kernel),
            STGCNBlock(64, 128, k_num, t_kernel, stride=2),
            STGCNBlock(128, 128, k_num, t_kernel),
            STGCNBlock(128, 256, k_num, t_kernel, stride=2),
            STGCNBlock(256, 256, k_num, t_kernel),
        ])
        self.temporal_transformer = TemporalTransformer(
            d_model=256, nhead=transformer_heads,
            num_layers=transformer_layers, dim_ff=transformer_ff, dropout=dropout,
        )
        self.head = build_head(self.head_type, 256, num_class=num_class, dropout=dropout)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)
        for blk in self.blocks:
            x = blk(x, self.A)
        x = x.mean(dim=3)                      # (N, 256, T')  rata-rata spasial
        x = x.permute(0, 2, 1).contiguous()    # (N, T', 256)
        x = self.temporal_transformer(x)
        x = x.mean(dim=1)                      # (N, 256) global temporal pooling
        out = self.head(x)
        if isinstance(out, dict):
            return out
        if self.head_type == "va":
            return {"va": out}
        return {"logits": out}


if __name__ == "__main__":
    for ht in ("va", "class", "dual"):
        m = STGCN_GraphTransformer(head_type=ht)
        y = m(torch.randn(2, 2, 150, 68))
        print(ht, {k: tuple(v.shape) for k, v in y.items()})
