# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Simplified phasor RMS small-signal analysis entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import VeraGridEngine.api as vge

from small_signal_analysis_main import DEFAULT_GRID, load_grid, set_models


def run_small_signal_analysis(grid) -> dict:
    """Run power flow and small-signal analysis with ``RmsProblemPhasor``."""
    pf_results = vge.power_flow(grid, vge.PowerFlowOptions(tolerance=1e-5))
    if not pf_results.converged:
        raise RuntimeError("Power flow did not converge")

    rms_options = vge.RmsOptions(
        time_step=0.01,
        simulation_time=1.0,
        tolerance=1e-6,
        max_iter=20,
    )
    problem = vge.RmsProblemPhasor(grid=grid, options=rms_options, pf_results=pf_results)

    ss_options = vge.RmsSmallSignalStabilityOptions(ss_assessment_time=0, verbose=0)
    ss_options.k = problem.get_states_number() + problem.get_diff_var_number()
    driver = vge.SmallSignalStabilityRmsDriver(
        grid=vge.MultiCircuit(Sbase=grid.Sbase),
        rms_options=rms_options,
        sss_options=ss_options,
        pf_results=pf_results,
    )
    driver.problem = problem
    driver.k = ss_options.k
    driver.run()

    eigenvalues = driver.results.eigenvalues
    finite = eigenvalues[np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1e6)]
    stable = bool(np.all(np.real(finite) <= 0.0)) if len(finite) else False
    margin = float(np.max(np.real(finite))) if len(finite) else float("nan")

    print(f"RmsProblemPhasor states={problem.get_states_number()} diff_vars={problem.get_diff_var_number()}")
    print(f"Finite eigenvalues={len(finite)} stable={stable} margin={margin:.6e}")

    return {
        "problem": problem,
        "pf_results": pf_results,
        "eigenvalues": finite,
        "participation_factors": driver.results.participation_factors,
        "stable": stable,
        "margin": margin,
    }


def plot_results(results: dict, output_path: Path | None = None) -> None:
    """Plot and save finite phasor eigenvalues."""
    eigenvalues = results["eigenvalues"]
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "small_signal_phasor_main.png"

    fig, ax = plt.subplots(figsize=(9, 6))
    if len(eigenvalues):
        colors = np.real(eigenvalues) > 0.0
        ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), c=colors, cmap="coolwarm", s=45, alpha=0.85)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="k", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(f"RMS Phasor Small-Signal Eigenvalues, margin={results['margin']:.3e}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {output_path}")


def main() -> None:
    grid_filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRID
    grid = load_grid(grid_filename)
    set_models(grid)
    results = run_small_signal_analysis(grid)
    plot_results(results)


if __name__ == "__main__":
    main()
