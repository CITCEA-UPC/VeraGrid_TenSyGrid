"""Locate the external STAMP_Public checkout used by the STAMP demos."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_stamp_public_root() -> Path:
    """Return a usable STAMP_Public checkout or raise an actionable error."""
    candidates = []
    configured = os.environ.get("STAMP_PUBLIC_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())

    repository_root = Path(__file__).resolve().parents[4]
    candidates.extend(
        (
            repository_root.parent / "STAMP_Public",
            Path.cwd() / "STAMP_Public",
        )
    )
    for candidate in candidates:
        if (candidate / "veragrid_stamp").is_dir():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise ModuleNotFoundError(
        "STAMP demos require a STAMP_Public checkout. Set STAMP_PUBLIC_ROOT "
        f"to its directory. Searched: {searched}"
    )


def add_stamp_public_to_path() -> Path:
    """Make STAMP_Public importable and return its resolved root."""
    root = find_stamp_public_root()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root
