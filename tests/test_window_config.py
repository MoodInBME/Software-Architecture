"""Panjang jendela jalur wajah sama di config pelatihan dan config runtime cermin."""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _yaml(rel):
    with open(REPO / rel, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_jendela_90_geser_15_sama_di_pelatihan_dan_runtime():
    st = _yaml("models/st_gcn_graph_tran/config.yaml")["data"]
    rt = _yaml("config/default.yaml")
    assert st["window_frames"] == rt["clip_len_stgcn"] == 90
    assert st["stride_frames"] == rt["stride"] == 15


def test_nilai_bawaan_kode_sama_dengan_config():
    src = (REPO / "inference.py").read_text(encoding="utf-8")
    assert 'c["clip_len_stgcn"] = 90' in src and 'c["stride"] = 15' in src
    tr = (REPO / "models/st_gcn_graph_tran/train.py").read_text(encoding="utf-8")
    assert '"window_frames": 90' in tr
