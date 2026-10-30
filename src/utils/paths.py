"""Resolve paths independently of the working directory."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_root():
    """TP2_DATA_DIR points to a directory containing task subdirectories."""
    return Path(os.environ.get("TP2_DATA_DIR", PROJECT_ROOT / "data" / "raw")).expanduser().resolve()


def task_files(task, root=None):
    """Case-insensitive official CSV names, with ambiguity rejected."""
    folder = (Path(root) if root is not None else data_root()) / task
    files = {}
    for role, stem in (("train", "train"), ("test", "test"), ("sample", "samplesubmission")):
        matches = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.csv' and p.stem.lower() == stem] if folder.is_dir() else []
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {task}/{role}: {[p.name for p in matches]}")
        files[role] = matches[0] if matches else None
    return files
