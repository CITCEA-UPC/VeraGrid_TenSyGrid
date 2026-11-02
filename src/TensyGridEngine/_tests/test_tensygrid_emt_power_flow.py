"""Regression checks for saved IEEE 9 EMT grids used by TenSyGrid."""

from pathlib import Path

import pytest

from TensyGridEngine.emt.emt_ieee9 import build_power_flow_options
from TensyGridEngine.emt.emt_power_flow import run_emt_power_flows
from VeraGridEngine.IO.file_open import FileOpen


GRID_DIR = Path(__file__).resolve().parents[3] / "Grids_and_profiles" / "grids"


@pytest.mark.parametrize("filename", [
    "IEEE 9 Bus EMT.veragrid",
    "IEEE 9 Bus EMT Multilinear.veragrid",
])
def test_saved_ieee9_emt_grid_uses_balanced_operating_point_when_sequence_data_invalid(filename):
    grid = FileOpen(str(GRID_DIR / filename)).open()
    balanced, three_phase = run_emt_power_flows(grid, build_power_flow_options())

    assert bool(balanced.converged)
    assert three_phase is None
