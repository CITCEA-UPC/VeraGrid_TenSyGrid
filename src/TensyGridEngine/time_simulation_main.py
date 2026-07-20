# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Simple RMS multilinear time-domain simulation for the TenSyGrid example."""

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import VeraGridEngine.api as vge
from VeraGridEngine.Simulations.Rms.numerical.back_euler_fx import BackEulerImplicitIntegration

from small_signal_analysis_main import DEFAULT_GRID, load_grid, set_models


def ensure_problem_event_api(problem) -> None:
    """Provide no-op event hooks expected by the BackEuler solver."""
    if not hasattr(problem, "get_next_forced_event_time"):
        def _get_next_forced_event_time(self, t_local_prev, t_macro_target):
            return None
        problem.get_next_forced_event_time = MethodType(_get_next_forced_event_time, problem)

    if not hasattr(problem, "update"):
        def _update(self, t, x_snapshot, variable_parameters):
            return None
        problem.update = MethodType(_update, problem)


def run_time_simulation(grid, t_end: float = 0.2, h: float = 0.01) -> dict:
    """Build an ``RmsProblemMultilinear`` and run a Backward Euler simulation."""
    pf_results = vge.power_flow(grid, vge.PowerFlowOptions(tolerance=1e-5))
    if not pf_results.converged:
        raise RuntimeError("Power flow did not converge")

    rms_options = vge.RmsOptions(
        time_step=h,
        simulation_time=t_end,
        tolerance=1e-6,
        max_iter=20,
        problem_type=vge.RmsProblemTypes.Multilinear,
    )
    problem = vge.RmsProblemMultilinear(grid=grid, options=rms_options, pf_results=pf_results)
    ensure_problem_event_api(problem)

    solver = BackEulerImplicitIntegration(
        problem=problem,
        t0=0.0,
        t_end=t_end,
        h=h,
        max_iter=rms_options.max_iter,
        tolerance=rms_options.tolerance,
    )
    t, y, well_initialized, converged = solver.simulate()
    print(
        f"Time simulation finished: steps={len(t)} "
        f"well_initialized={well_initialized} converged={converged} final_norm={np.linalg.norm(y[-1, :]):.6e}"
    )
    return {
        "problem": problem,
        "pf_results": pf_results,
        "t": t,
        "y": y,
        "well_initialized": well_initialized,
        "converged": converged,
    }


def _select_series(problem, y: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variables = list(getattr(problem, "state_and_algebraic_vars", []))
    preferred_names = [
        "omega",
        "delta",
        "Vf",
        "Pg",
        "Qg",
        "Vr_",
        "Vi_",
        "Irg",
        "Iig",
    ]
    selected: list[tuple[str, np.ndarray]] = []

    for prefix in preferred_names:
        for i, var in enumerate(variables):
            name = str(getattr(var, "name", var))
            if i >= y.shape[1] or not name.startswith(prefix):
                continue
            if name.startswith("Vr_aux") or name.startswith("Vi_aux"):
                continue
            selected.append((name, y[:, i]))
            break

    if selected:
        return selected[:6]

    n_series = min(6, y.shape[1])
    return [(str(variables[i]) if i < len(variables) else f"y[{i}]", y[:, i]) for i in range(n_series)]


def plot_timeseries(results: dict, output_path: Path | None = None) -> None:
    """Plot selected time-domain variables."""
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "time_simulation_main.png"

    t = results["t"]
    y = results["y"]
    problem = results["problem"]
    series = _select_series(problem, y)

    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 1.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, series):
        ax.plot(t, values, linewidth=1.7)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("RMS Multilinear Time Simulation")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved timeseries plot: {output_path}")


def main() -> None:
    grid_filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRID
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

    grid = load_grid(grid_filename)
    set_models(grid)
    results = run_time_simulation(grid=grid, t_end=t_end, h=h)
    plot_timeseries(results)


if __name__ == "__main__":
    main()
