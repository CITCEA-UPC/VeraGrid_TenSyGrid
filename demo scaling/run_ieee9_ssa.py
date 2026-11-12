#!/usr/bin/env python3
"""Run exploratory EMT and RMS SSA for the saved IEEE-9 grid.

The archive contains EMT blocks but no RMS blocks. The RMS helper constructs
different phasor generator, load, and line models. Its spectrum is therefore
not an equivalent-model validation of the saved EMT spectrum.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/veragrid-demo-scaling-mpl")))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path("/tmp/veragrid-demo-scaling-numba")))
os.environ.setdefault("VERAGRID_EMT_INIT_CACHE_DIR", str(Path("/tmp/veragrid-demo-scaling-emt-init")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt.emt_ieee9_small_signal import run_small_signal_analysis
from TensyGridEngine.small_signal_analysis_main import set_models, run_small_signal_analysis as run_rms
from VeraGridEngine import api as vge

GRID = ROOT / "Grids_and_profiles/grids/IEEE 9 Bus EMT.veragrid"
OUT = Path(__file__).resolve().parent


def save_modes(name: str, values: np.ndarray, elapsed: float) -> None:
    values = np.asarray(values)
    with (OUT / f"{name}_modes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "real", "imag", "magnitude"))
        for index, value in enumerate(values):
            writer.writerow((index, value.real, value.imag, abs(value)))
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(values.real, values.imag, s=30)
    if name.startswith("emt"):
        theta = np.linspace(0, 2 * np.pi, 500)
        ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1, label="unit circle")
        ax.set(xlabel="Re(mu)", ylabel="Im(mu)", title="IEEE-9 EMT Floquet multipliers")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
    else:
        ax.axvline(0, color="black", linestyle="--", lw=1)
        ax.set(xlabel="Re(lambda) [1/s]", ylabel="Im(lambda) [rad/s]",
               title="IEEE-9 RMS eigenvalues")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / f"{name}_ssa.png", dpi=180)
    plt.close(fig)
    print(f"{name}: {len(values)} modes, {elapsed:.2f} s, plot={OUT / f'{name}_ssa.png'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("emt", "rms"))
    parser.add_argument("--emt-modes", default="12",
                        help="Number of dominant EMT multipliers, or 'all' for the dense full spectrum")
    args = parser.parse_args()
    if not GRID.exists():
        raise FileNotFoundError(GRID)
    if args.only != "rms":
        n_modes = None if args.emt_modes == "all" else int(args.emt_modes)
        if n_modes is not None and n_modes < 1:
            parser.error("--emt-modes must be a positive integer or 'all'")
        start = perf_counter()
        result = run_small_signal_analysis(grid_path=GRID, n_modes=n_modes)
        name = "emt_all" if args.emt_modes == "all" else "emt"
        save_modes(name, result.multipliers, perf_counter() - start)
    if args.only != "emt":
        print("RMS uses generated phasor models; these are not dynamically equivalent to the saved EMT blocks.", flush=True)
        start = perf_counter()
        grid = vge.open_file(str(GRID))
        set_models(grid)
        result = run_rms(grid)
        save_modes("rms", result["eigenvalues"], perf_counter() - start)


if __name__ == "__main__":
    main()
