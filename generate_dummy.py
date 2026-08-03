"""
MoodIn - Generate Dummy Weights (Pseudo-Pretrained)

Menghasilkan bobot model yang bukan sekadar inisialisasi acak PyTorch.
Strategi: jalankan beberapa iterasi training loop pada data sintetis
sehingga distribusi bobot lebih menyerupai model terlatih (non-uniform,
konvergen sebagian). Output model ini tetap tidak menghasilkan prediksi
bermakna, tapi secara statistik lebih realistis untuk demo.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.st_gcn_graph_tran.model import STGCN_GraphTransformer
from models.stc_rppg.model import STCrPPG

WEIGHT_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(WEIGHT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WARMUP = 50  # jumlah iterasi "pemanasan" untuk membuat bobot non-trivial


def warmup_stgcn():
    """Latih ST-GCN beberapa iterasi pada data sintetis."""
    print("[ST-GCN] Inisialisasi model...")
    model = STGCN_GraphTransformer(
        in_channels=2, num_class=5, num_node=68, graph_strategy="spatial",
        max_hop=1, t_kernel=9, transformer_layers=2, transformer_heads=8,
        transformer_ff=512, dropout=0.3, head_type="va"
    ).to(DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"[ST-GCN] Melakukan {NUM_WARMUP} iterasi pemanasan pada data sintetis...")
    for i in range(NUM_WARMUP):
        # Input sintetis: (batch=1, C=2, T=150, V=68)
        x = torch.randn(1, 2, 150, 68, device=DEVICE)
        # Target VA: valence dan arousal acak di [-1, 1]
        target_va = torch.tanh(torch.randn(1, 2, device=DEVICE))

        out = model(x)
        va_pred = out["va"]
        loss = loss_fn(va_pred, target_va)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"  Iterasi {i+1}/{NUM_WARMUP} | Loss: {loss.item():.4f}")

    model.eval()
    path = os.path.join(WEIGHT_DIR, "dummy_stgcn.pth")
    torch.save(model.state_dict(), path)
    print(f"[ST-GCN] Bobot tersimpan: {path} ({os.path.getsize(path) / 1e6:.1f} MB)\n")


def warmup_stcrppg():
    """Latih STC-rPPG beberapa iterasi pada data sintetis."""
    print("[STC-rPPG] Inisialisasi model...")
    model = STCrPPG().to(DEVICE)
    model.train()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    loss_fn = nn.MSELoss()

    print(f"[STC-rPPG] Melakukan {NUM_WARMUP} iterasi pemanasan pada data sintetis...")
    for i in range(NUM_WARMUP):
        # Input sintetis: (batch=1, C=3, T=32, H=32, W=32)
        # Ukuran kecil supaya bisa berjalan di CPU laptop dalam hitungan menit.
        # Arsitektur tetap sama; hanya ukuran tensor latihan yang diperkecil.
        x = torch.rand(1, 3, 32, 32, 32, device=DEVICE)
        
        out = model(x)  # output: (1, T')
        T_out = out.shape[-1]
        # Target: sinyal rPPG sintetis (gelombang sinus ~1.2 Hz = 72 BPM)
        t = torch.linspace(0, T_out / 30.0, T_out, device=DEVICE)
        target = torch.sin(2 * np.pi * 1.2 * t).unsqueeze(0).expand(1, -1)

        loss = loss_fn(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"  Iterasi {i+1}/{NUM_WARMUP} | Loss: {loss.item():.4f}")

    model.eval()
    path = os.path.join(WEIGHT_DIR, "dummy_stcrppg.pth")
    torch.save(model.state_dict(), path)
    print(f"[STC-rPPG] Bobot tersimpan: {path} ({os.path.getsize(path) / 1e6:.1f} MB)\n")


if __name__ == "__main__":
    print("=" * 55)
    print(" MoodIn - Generate Pseudo-Pretrained Weights ".center(55, "="))
    print("=" * 55)
    print(f"Device: {DEVICE}\n")
    warmup_stgcn()
    warmup_stcrppg()
    print("Selesai. Bobot dummy pseudo-pretrained tersimpan di weights/")
