"""Rectangular-interface variant of the full dynamic STAMP RMS case."""
from __future__ import annotations

from .wscc_case import build_stamp_wscc_grid


def build_stamp_phasor_rms_grid():
    """Build STAMP's dynamic lines, bus capacitors, RL loads, and sources."""
    grid = build_stamp_wscc_grid(
        dynamic_lines=True,
        full_dynamic_network=True,
        nonlinear_converters=True,
        phasor_network=True,
    )
    grid.name = "STAMP WSCC SG-GFOR-GFOL full dynamic phasor RMS"
    return grid
