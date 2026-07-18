"""
ST-GCN + Graph-Transformer untuk klasifikasi mood MoodIn (5 kelas).

Alur:
  input (N, C, T, V) landmark wajah
    -> data BatchNorm
    -> tumpukan STGCNBlock (ekstraksi fitur spatio-temporal graf)
    -> pooling spasial (rata-rata node)
    -> Temporal Transformer (dependensi temporal jangka panjang)
    -> global temporal pooling
    -> classifier (5 kelas: Ceria/Cemas/Tertekan/Rileks/Netral)

Catatan: butuh PyTorch. Belum dieksekusi di sandbox (torch tidak terpasang
& jaringan mati) -- sudah lulus py_compile untuk validasi sintaks.
"""
import torch
import torch.nn as nn

try:
    from .graph import FaceGraph
    from .st_gcn_block import STGCNBlock
    from .transformer import TemporalTransformer
except ImportError:  # dijalankan sebagai skrip lepas
    from graph import FaceGraph
    from st_gcn_block import STGCNBlock
    from transformer import TemporalTransformer


class STGCN_GraphTransformer(nn.Module):
    def __init__(self, in_channels=2, num_class=5, num_node=68,
                 graph_strategy="spatial", max_hop=1, t_kernel=9,
                 transformer_layers=2, transformer_heads=8, dropout=0.3):
        super().__init__()
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
            num_layers=transformer_layers, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_class)

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
        x = self.dropout(x)
        return self.fc(x)


if __name__ == "__main__":
    model = STGCN_GraphTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameter: {n_params/1e6:.2f} M")
    dummy = torch.randn(2, 2, 150, 68)  # (N, C, T, V)
    out = model(dummy)
    print("Bentuk output:", out.shape)  # harapan: (2, 5)
