"""Utilitas bersama skrip validasi: semua keluaran wajib di luar repo (bawaan D:/moodin_runs)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def pastikan_di_luar_repo(path: str) -> Path:
    """Path keluaran yang sudah dibuat foldernya. Galat bila berada di dalam repo."""
    p = Path(path).expanduser().resolve()
    if p == REPO or p.is_relative_to(REPO):
        raise SystemExit(
            f"Keluaran {p} berada di dalam repo; simpan di D:/moodin_runs/."
        )
    (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def info_lingkungan() -> dict:
    """Ringkasan versi untuk dicatat di setiap hasil (tanpa data pribadi)."""
    d = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "waktu": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        import torch

        d["torch"] = torch.__version__
        d["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        d["torch"] = None
    try:
        d["git"] = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        d["git"] = None
    return d


def tulis_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
