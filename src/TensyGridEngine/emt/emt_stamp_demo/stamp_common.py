"""Locate the STAMP reference sources bundled with this repository."""

from __future__ import annotations

import sys
from pathlib import Path


def find_stamp_public_root() -> Path:
    """Return this repository's source directory containing veragrid_stamp."""
    source_root = Path(__file__).resolve().parents[3]
    if not (source_root / "veragrid_stamp").is_dir():
        raise ModuleNotFoundError("Bundled veragrid_stamp sources are missing")
    return source_root


def add_stamp_public_to_path() -> Path:
    """Make the bundled STAMP sources importable for existing demos."""
    root = find_stamp_public_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root
