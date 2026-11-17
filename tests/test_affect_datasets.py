"""Uji sintetis untuk loader AFEW-VA, DISFA, OpenFace, metrik afek, Tabel 3.3, dan rugi bermask.

Tidak membutuhkan data asli, mediapipe, atau cv2. Jalankan dari akar repo:
    python -m pytest tests/test_affect_datasets.py -q
"""

import json
import math
import os

import numpy as np
import pytest
import torch

from data.loaders import afew_va, disfa
from data.loaders.affect_landmark_dataset import AffectLandmarkDataset
from data.loaders.openface_csv import read_openface_csv
from data.loaders.subject_split import k_fold_subjects, split_subjects
from data.preprocessing import affect_windows as W
from evaluation.affect_metrics import ccc, icc31, nme
from models.st_gcn_graph_tran.au_va_heads import AUVAHead, masked_multitask_loss
from models.st_gcn_graph_tran.labels_tabel33 import sam_to_mood, va_to_mood_tabel33


def _face(seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (68, 2)).astype(np.float32)
    pts[36], pts[45] = (-1.0, 0.0), (1.0, 0.0)
    return pts


# ---------------------------------------------------------------- affect_windows
def test_normalize_scale_invariant():
    f = _face()
    seq = np.stack([f, f * 3 + 7])
    out = W.normalize_landmarks_seq(seq)
    np.testing.assert_allclose(out[0], out[1], atol=1e-5)
    d = np.linalg.norm(out[:, 36] - out[:, 45], axis=-1)
    np.testing.assert_allclose(d, 1.0, atol=1e-4)


def test_fill_missing_and_all_missing():
    seq = np.stack(
        [np.full((68, 2), np.nan), _face(1), np.full((68, 2), np.nan)]
    ).astype(np.float32)
    out, valid = W.fill_missing(seq)
    assert valid.tolist() == [False, True, False]
    np.testing.assert_array_equal(out[0], out[1])
    np.testing.assert_array_equal(out[2], out[1])
    with pytest.raises(ValueError):
        W.fill_missing(np.full((3, 68, 2), np.nan, np.float32))


def test_resample_20_to_30_preserves_duration_and_line():
    t = np.arange(40, dtype=np.float32)  # 2 detik pada 20 fps
    out = W.resample_time(t, 20.0, 30.0)
    assert len(out) == int(np.floor(39 / 20 * 30)) + 1
    np.testing.assert_allclose(out, np.arange(len(out)) * 20 / 30, atol=1e-5)


def test_resample_keeps_nan_without_poisoning_exact_points():
    v = np.array([0.0, np.nan, 2.0, 3.0], np.float32)
    out = W.resample_time(v, 1.0, 2.0)
    assert out[0] == 0.0 and np.isnan(out[1]) and np.isnan(out[2]) and out[4] == 2.0


def test_chunk_indices_matches_repo_semantics():
    assert W.chunk_indices(50, 90, 15) == [(0, 50)]
    idx = W.chunk_indices(200, 90, 15)
    assert idx[0] == (0, 90) and idx[-1] == (110, 200)


def test_make_clips_masks_and_drops_unlabeled():
    T = 120
    lm = np.repeat(_face()[None], T, 0)
    valid = np.ones(T, bool)
    va = np.full((T, 2), np.nan, np.float32)
    va[:90] = 0.5
    recs = W.make_clips(lm, valid, "x", "s1", "afew-va", 90, 30, va=va, n_au=12)
    assert recs[0].va_mask == 1.0 and math.isclose(recs[0].valence, 0.5)
    assert all(r.au_mask == [0.0] * 12 for r in recs)
    assert all(
        r.va_mask == 1.0 for r in recs
    )  # jendela separuh berlabel tetap valid (rerata NaN-aware)
    none = np.full((T, 2), np.nan, np.float32)
    assert W.make_clips(lm, valid, "y", "s1", "afew-va", 90, 30, va=none) == []


# ---------------------------------------------------------------- AFEW-VA
def _write_afew_clip(root, cid, n=40, actor="A1", v=5, a=-10):
    d = os.path.join(root, cid)
    os.makedirs(d)
    frames = {
        f"{i:05d}": {"valence": v, "arousal": a, "landmarks": _face(i).tolist()}
        for i in range(n)
    }
    with open(os.path.join(d, f"{cid}.json"), "w") as f:
        json.dump({"actor": actor, "video_id": cid, "frames": frames}, f)


def test_afew_read_scales_va(tmp_path):
    _write_afew_clip(tmp_path, "001")
    clip = afew_va.read_clip_json(str(tmp_path / "001" / "001.json"))
    assert clip["landmarks_gt"].shape == (40, 68, 2)
    np.testing.assert_allclose(clip["va"][0], [0.5, -1.0])
    assert clip["actor"] == "A1"


def test_afew_rejects_bad_structure(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"video": 1}))
    with pytest.raises(KeyError):
        afew_va.read_clip_json(str(p))


def test_afew_prepare_gt_end_to_end(tmp_path):
    root, out = tmp_path / "afew", tmp_path / "prep"
    os.makedirs(root)
    _write_afew_clip(root, "001", n=100, actor="A1")
    _write_afew_clip(root, "002", n=30, actor="A2", v=-4, a=4)
    man = afew_va.prepare(str(root), str(out), "gt", 90, 15, 25.0, 30.0, 0.2)
    assert len(man) >= 2
    ds = AffectLandmarkDataset(
        str(out / "manifest.json"), au_list=disfa.DISFA_AUS, clip_len=90
    )
    x, va, vm, au, am = ds[0]
    assert x.shape == (2, 90, 68) and vm.item() == 1.0 and am.sum().item() == 0
    assert set(ds.subjects()) == {"A1", "A2"}


def test_download_refuses_without_agreement(tmp_path):
    with pytest.raises(SystemExit):
        afew_va.download(str(tmp_path), agree=False)


# ---------------------------------------------------------------- DISFA
def test_disfa_read_au_file(tmp_path):
    p = tmp_path / "SN001_au1.txt"
    p.write_text("1,0\n2,3\n4,5\n")
    v = disfa.read_au_file(str(p))
    assert v.tolist()[:2] == [0.0, 3.0] and np.isnan(v[2]) and v[3] == 5.0
    p.write_text("1,7\n")
    with pytest.raises(ValueError):
        disfa.read_au_file(str(p))


def test_disfa_subject_labels_and_missing_au(tmp_path):
    d = tmp_path / "ActionUnit_Labels" / "SN001"
    os.makedirs(d)
    for au in disfa.DISFA_AUS:
        (d / f"SN001_au{au}.txt").write_text(
            "".join(f"{i},{au % 6}\n" for i in range(1, 11))
        )
    lab = disfa.read_subject_labels(str(d))
    assert lab.shape == (10, 12) and lab.max() <= 1.0
    os.remove(d / "SN001_au26.txt")
    with pytest.raises(FileNotFoundError):
        disfa.read_subject_labels(str(d))


def test_dataset_maps_au_by_number(tmp_path):
    out = tmp_path / "prep"
    T = 60
    recs = W.make_clips(
        np.repeat(_face()[None], T, 0),
        np.ones(T, bool),
        "d",
        "SN001",
        "disfa",
        30,
        30,
        au=np.full((T, 12), 0.4, np.float32),
    )
    man = []
    W.save_clips(recs, str(out / "clips"), man)
    (out / "manifest.json").write_text(json.dumps(man))
    (out / "meta.json").write_text(json.dumps({"au_list": disfa.DISFA_AUS}))
    ds = AffectLandmarkDataset(
        str(out / "manifest.json"), au_list=[12, 7, 1], clip_len=30
    )
    _, _, vm, au, am = ds[0]
    assert vm.item() == 0.0
    assert am.tolist() == [1.0, 0.0, 1.0]  # AU7 tidak ada di DISFA
    assert abs(au[0].item() - 0.4) < 1e-6


# ---------------------------------------------------------------- OpenFace
def test_openface_csv(tmp_path):
    cols = ["frame", " face_id", " timestamp", " confidence", " success"]
    cols += (
        [f" x_{i}" for i in range(68)]
        + [f" y_{i}" for i in range(68)]
        + [" AU01_r", " AU12_r"]
    )
    rows = [
        [1, 0, 0.0, 0.98, 1] + [1.0] * 136 + [2.0, 3.0],
        [2, 0, 0.05, 0.3, 1] + [1.0] * 136 + [1.0, 1.0],
    ]
    p = tmp_path / "of.csv"
    p.write_text(",".join(cols) + "\n" + "\n".join(",".join(map(str, r)) for r in rows))
    r = read_openface_csv(str(p))
    assert r["landmarks"].shape == (2, 68, 2)
    assert r["success"].tolist() == [True, False]
    assert r["au"][12][0] == 3.0 and np.isnan(r["au"][1][1])


# ---------------------------------------------------------------- metrik
def test_metrics_reference_values():
    gt = np.repeat(_face()[None], 2, 0)
    assert np.allclose(nme(gt, gt), 0)
    shifted = gt + np.array([0.2, 0.0], np.float32)
    np.testing.assert_allclose(
        nme(shifted, gt), 0.1, atol=1e-6
    )  # 0,2 dibagi jarak mata luar 2
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(icc31(y, y), 1.0)
    assert math.isclose(
        icc31(y, y + 1.0), 1.0
    )  # konsistensi: bias konstan tidak dihukum
    assert math.isclose(ccc(y, y), 1.0)
    assert ccc(y, y + 1.0) < 1.0  # CCC menghukum bias


# ---------------------------------------------------------------- Tabel 3.3
@pytest.mark.parametrize(
    "v,a,exp",
    [
        (5, 5, "Netral"),
        (4, 6, "Netral"),
        (7, 7, "Ceria"),
        (7, 5, "Rileks"),
        (7, 4, "Rileks"),
        (3, 7, "Cemas"),
        (3, 3, "Tertekan"),
        (3, 5, "Tertekan"),
        (5, 9, None),
        (5, 1, None),
        (9, 6, "Ceria"),
    ],
)
def test_tabel33(v, a, exp):
    assert sam_to_mood(v, a) == exp


def test_tabel33_differs_from_gemastik_circle():
    # (0,22; 0,22): radius 0,31 > 0,2 sehingga lingkaran lama memberi Ceria, Tabel 3.3 memberi Netral
    assert va_to_mood_tabel33(0.22, 0.22) == "Netral"


# ---------------------------------------------------------------- rugi bermask
def test_masked_loss_mixed_batch_and_empty_parts():
    torch.manual_seed(0)
    head = AUVAHead(16, n_au=12)
    params = [p for p in head.parameters() if p.requires_grad]
    out = head(torch.randn(4, 16))
    va = torch.tensor([[0.5, 0.1], [-0.2, 0.3], [0.0, 0.0], [0.0, 0.0]])
    va_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    au = torch.rand(4, 12)
    au_mask = torch.zeros(4, 12)
    au_mask[2:] = 1.0
    loss, parts = masked_multitask_loss(out, va, va_mask, au, au_mask)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    assert torch.isfinite(loss) and parts["au"] > 0 and parts["va"] > 0
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    out2 = head(torch.randn(2, 16))
    loss2, parts2 = masked_multitask_loss(
        out2, torch.zeros(2, 2), torch.zeros(2), torch.zeros(2, 12), torch.zeros(2, 12)
    )
    grads2 = torch.autograd.grad(loss2, params, allow_unused=True)
    assert loss2.item() == 0.0 and parts2 == {"au": 0.0, "va": 0.0}
    assert all(g is None or torch.isfinite(g).all() for g in grads2)


def test_splits_are_disjoint():
    subs = [f"SN{i:03d}" for i in range(27)]
    sp = split_subjects(subs)
    assert not (set(sp["train"]) & set(sp["test"])) and not (
        set(sp["val"]) & set(sp["test"])
    )
    assert sum(len(v) for v in sp.values()) == 27
    folds = k_fold_subjects(subs, k=3)
    assert sorted(s for f in folds for s in f["test"]) == sorted(subs)
