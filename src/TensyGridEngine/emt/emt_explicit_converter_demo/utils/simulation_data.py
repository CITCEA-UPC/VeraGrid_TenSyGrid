"""Consistent MATLAB-readable output for the four explicit EMT demos."""

from pathlib import Path

import numpy as np
from scipy.io import savemat


DEMO_DIR = Path(__file__).resolve().parents[1]


def case_dir(name: str) -> Path:
    """Return ``simulation_data/<demo>`` beside the runnable scripts."""
    folder = DEMO_DIR / "simulation_data" / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_case_data(folder: Path, *, S, Phi, time, states, state_names,
                   inputs, input_names, metadata=None) -> None:
    """Save aligned, named traces and the fixed-parameter S/Phi matrices.

    Each trace matrix has one row per time sample; the matching names are
    MATLAB cell arrays. ``metadata`` is stored with the traces for provenance.
    """
    folder.mkdir(parents=True, exist_ok=True)
    time = np.asarray(time, dtype=float).reshape(-1, 1)
    states = np.asarray(states, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    if states.shape != (len(time), len(state_names)):
        raise ValueError("State trace dimensions do not match time and names")
    if inputs.shape != (len(time), len(input_names)):
        raise ValueError("Input trace dimensions do not match time and names")
    common = dict(metadata or {})
    savemat(folder / "S.mat", {"S": S}, do_compression=True)
    savemat(folder / "Phi.mat", {"Phi": Phi}, do_compression=True)
    savemat(folder / "inputs.mat", {
        "time": time, "inputs": inputs,
        "input_names": np.asarray(input_names, dtype=object).reshape(1, -1),
        **common,
    }, do_compression=True)
    savemat(folder / "states.mat", {
        "time": time, "states": states,
        "state_names": np.asarray(state_names, dtype=object).reshape(1, -1),
        **common,
    }, do_compression=True)
