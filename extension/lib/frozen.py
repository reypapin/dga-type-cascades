"""Frozen configuration for preregistration v2 (extension/config/frozen_v2.json)."""

from __future__ import annotations

import json
from pathlib import Path

FROZEN = Path(__file__).resolve().parents[1] / "config" / "frozen_v2.json"


def update_frozen(block: str, payload: dict) -> Path:
    data = json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else {}
    data[block] = payload
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return FROZEN
