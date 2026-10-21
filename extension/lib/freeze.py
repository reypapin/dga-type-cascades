"""Freeze files by SHA-256 and make the test analysis run only once."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_freeze(sums_path: Path, files: list[Path]) -> None:
    base = Path(sums_path).parent
    lines = [f"{sha256(f)}  {os.path.relpath(Path(f).resolve(), base.resolve())}" for f in files]
    Path(sums_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_freeze(sums_path: Path) -> None:
    base = Path(sums_path).parent
    for line in Path(sums_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        actual = sha256(base / rel)
        if actual != expected:
            raise SystemExit(f"{rel} changed after the freeze ({actual} != {expected})")


def guard_single_run(summary_path: Path) -> None:
    if Path(summary_path).exists():
        raise SystemExit(f"{summary_path} already exists. The test runs once; refusing to run again.")
