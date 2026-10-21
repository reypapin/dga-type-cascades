"""Load the saved routers (extension/models/), checking each file against MANIFEST.json before unpickling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_router(name: str):
    # Our own routers (scripts/save_artifacts.py). joblib unpickles, so the file must match its SHA-256 first.
    manifest = json.loads((MODELS_DIR / "MANIFEST.json").read_text())
    path = MODELS_DIR / f"{name}.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][path.name]:
        raise SystemExit(f"{path.name} does not match extension/models/MANIFEST.json")
    return joblib.load(path)
