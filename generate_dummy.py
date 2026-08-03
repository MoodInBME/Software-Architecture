import os
import torch
import sys

# Tambahkan root ke sys.path agar impor models berhasil
sys.path.insert(0, os.path.abspath(r"C:\Users\Jeremia\GEMASTIK\MoodIn_Software"))

try:
    from models.st_gcn_graph_tran.model import STGCN_GraphTransformer
    from models.stc_rppg.model import STCrPPG
    
    os.makedirs(r"C:\Users\Jeremia\GEMASTIK\MoodIn_Software\weights", exist_ok=True)
    
    # Buat ST-GCN
    print("Membuat model ST-GCN...")
    stgcn = STGCN_GraphTransformer(
        in_channels=2, num_class=5, num_node=68, 
        graph_strategy="spatial", max_hop=1, t_kernel=9, 
        transformer_layers=2, transformer_heads=8, 
        transformer_ff=512, dropout=0.3, head_type="va"
    )
    torch.save(stgcn.state_dict(), r"C:\Users\Jeremia\GEMASTIK\MoodIn_Software\weights\dummy_stgcn.pth")
    print("ST-GCN dummy weights tersimpan.")
    
    # Buat STC-rPPG
    print("Membuat model STC-rPPG...")
    rppg = STCrPPG()
    torch.save(rppg.state_dict(), r"C:\Users\Jeremia\GEMASTIK\MoodIn_Software\weights\dummy_stcrppg.pth")
    print("STC-rPPG dummy weights tersimpan.")
    
except Exception as e:
    print(f"Error: {e}")
