# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""EMT Floquet small-signal analysis entry point for the TenSyGrid dashboard."""

from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import VeraGridEngine.api as vge
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
    SmallSignalStabilityEmtDriver,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import (
    SmallSignalStabilityEmtOptions,
)
from VeraGridEngine.enumerations import SmallSignalEmtBuildTypes
from TensyGridEngine.emt.emt_ieee9 import attach_emt_models, build_emt_options, build_power_flow_options


DEFAULT_GRID = "IEEE 9 Bus EMT.veragrid"
N_MODES = 20
MARGINAL_MULTIPLIER_TOLERANCE = 1.0e-4
STEPS_PER_PERIOD = 600
LIFTED_STEPS_PER_PERIOD = 1200
FORCE_MULTILINEAR_MACHINE = True
GOVERNOR_K = 40.0


def load_grid(grid_filename: str = DEFAULT_GRID):
    """Load a VeraGrid grid from ``Grids_and_profiles/grids``."""
    grid_path = project_base / "Grids_and_profiles" / "grids" / grid_filename
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    grid = vge.open_file(str(grid_path))
    print(f"Loaded grid: {grid_path}")
    return grid


def _has_emt_models(grid) -> bool:
    for collection in (grid.buses, grid.generators, grid.lines, grid.loads,
                       grid.transformers2w, grid.shunts):
        for device in collection:
            model = getattr(device, "emt_model", None)
            if model is not None and not model.empty():
                return True
    return False


def _is_lifted_machine(grid) -> bool:
    """Detect the trig-lifted (multilinear) EMT machine from its saved states."""
    for generator in grid.generators:
        model = getattr(generator, "emt_model", None)
        if model is None or model.empty():
            continue
        for child in model.children:
            for state in getattr(child, "state_vars", []):
                if state.name == "u_cos_sauer_pai":
                    return True
    return False


def run_emt_small_signal_analysis(grid):
    """Run power flow and EMT Floquet small-signal analysis."""
    if not _has_emt_models(grid):
        raise RuntimeError("The loaded grid does not contain EMT models")

    lifted_machine = _is_lifted_machine(grid)
    if FORCE_MULTILINEAR_MACHINE:
        attach_emt_models(
            grid,
            multilinear_machine=True,
            governor_k=GOVERNOR_K,
        )
        lifted_machine = True

    steps_per_period = LIFTED_STEPS_PER_PERIOD if lifted_machine else STEPS_PER_PERIOD
    period = 1.0 / float(grid.fBase)

    power_flow_options = build_power_flow_options()
    balanced_pf = PowerFlowDriver(grid=grid, options=power_flow_options)
    balanced_pf.run()
    three_phase_pf = PowerFlowDriver3Ph(grid=grid, options=power_flow_options)
    three_phase_pf.run()
    if not bool(balanced_pf.results.converged) or not bool(three_phase_pf.results.converged):
        raise RuntimeError("Power flow failed before EMT small-signal analysis")

    emt_options = build_emt_options()
    emt_options.time_step = period / float(steps_per_period)
    emt_options.simulation_time = period

    small_signal_options = SmallSignalStabilityEmtOptions(
        k=N_MODES,
        target_period=period,
        max_krylov_dim=max(30, 2 * N_MODES),
        ss_assessment_time=period,
        verbose=0,
        max_restarts=0,
        build_type=SmallSignalEmtBuildTypes.Arnoldi,
        consistent_dae_tangent=lifted_machine,
    )
    driver = SmallSignalStabilityEmtDriver(
        grid=grid,
        emt_options=emt_options,
        sss_options=small_signal_options,
        pf_results=balanced_pf.results,
        pf_results_3ph=three_phase_pf.results,
    )
    problem = driver.problem
    driver.run()
    if driver.results is None:
        raise RuntimeError("EMT small-signal analysis returned no results")

    eigenvalues = driver.results.eigenvalues
    finite = eigenvalues[np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1e6)]
    multipliers = np.asarray(driver.results.multipliers)
    finite_mu = multipliers[np.isfinite(multipliers)]
    rho = float(np.max(np.abs(finite_mu))) if len(finite_mu) else float("nan")
    margin = float(rho - 1.0) if len(finite_mu) else float("nan")
    stable = bool(rho <= 1.0 + MARGINAL_MULTIPLIER_TOLERANCE) if len(finite_mu) else False
    unstable_threshold = float(np.log(1.0 + MARGINAL_MULTIPLIER_TOLERANCE) / period)

    print(f"EMT machine={'multilinear' if lifted_machine else 'original'} "
          f"steps_per_period={steps_per_period}")
    print(f"EMT states={problem.get_states_number()} diff_vars={problem.get_diff_var_number()}")
    print(f"Finite eigenvalues={len(finite)} rho={rho:.9f} "
          f"stable={stable} margin={margin:.6e}")

    return {
        "problem": problem,
        "pf_results": balanced_pf.results,
        "results": driver.results,
        "eigenvalues": finite,
        "stable": stable,
        "margin": margin,
        "unstable_threshold": unstable_threshold,
        "period": period,
    }


def plot_results(results: dict, output_path: Path | None = None) -> None:
    """Plot and save the finite eigenvalues."""
    eigenvalues = results["eigenvalues"]
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "small_signal_analysis_main_emt.png"

    fig, ax = plt.subplots(figsize=(9, 6))
    if len(eigenvalues):
        colors = np.real(eigenvalues) > 0.0
        ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), c=colors, cmap="coolwarm", s=45, alpha=0.85)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="k", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("Real Part [1/s]")
    ax.set_ylabel("Imaginary Part [rad/s]")
    ax.set_title(f"EMT Floquet Eigenvalues, rho-1={results['margin']:.3e}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def _damping(re_: float, im: float) -> float:
    mag = np.sqrt(re_ * re_ + im * im)
    return -re_ / mag if mag > 1e-12 else 0.0


def _freq_hz(im: float) -> float:
    return abs(im) / (2 * np.pi)


def _hash_operating_point(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype=float).tobytes()).hexdigest()


def save_results(results: dict, grid, grid_filename: str, elapsed: float) -> None:
    eigenvalues = results["eigenvalues"]
    pf_results = results["pf_results"]
    emt_results = results["results"]
    problem = results["problem"]

    eigenvalues_list = [
        {"re": float(np.real(ev)), "im": float(np.imag(ev))}
        for ev in eigenvalues
    ]

    pf_matrix = emt_results.participation_factors
    state_var_names = [
        str(v.name) if hasattr(v, "name") else f"state_{i}"
        for i, v in enumerate(problem.get_state_vars())
    ]

    critical_modes = []
    for i, ev in enumerate(eigenvalues):
        re, im = float(np.real(ev)), float(np.imag(ev))
        if abs(im) <= 1.0e-9:
            continue
        damping = float(_damping(re, im))

        participation = {}
        if pf_matrix is not None and i < pf_matrix.shape[1]:
            pf_col = pf_matrix[:, i]
            for j, var_name in enumerate(state_var_names):
                if j < len(pf_col):
                    participation[str(var_name)] = float(pf_col[j])

        top_participation = dict(sorted(participation.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

        critical_modes.append({
            "mode": i,
            "re": re,
            "im": im,
            "damping": damping,
            "freq_hz": float(_freq_hz(im)),
            "critical": bool(abs(damping) < 0.05),
            "participation_factors": top_participation,
        })
    critical_modes.sort(key=lambda m: m["damping"])

    bus_voltages = []
    voltages = pf_results.voltage
    bus_names = pf_results.bus_names if hasattr(pf_results, "bus_names") else []
    bus_types = pf_results.bus_types if hasattr(pf_results, "bus_types") else []

    for i, bus in enumerate(grid.buses):
        v_complex = voltages[i] if i < len(voltages) else 1.0 + 0j
        vm = float(abs(v_complex))
        theta = float(np.angle(v_complex))
        bus_type = str(bus_types[i]) if i < len(bus_types) else "Unknown"
        bus_name = str(bus_names[i]) if i < len(bus_names) else str(bus.name)

        bus_voltages.append({
            "id": bus.idtag if hasattr(bus, "idtag") else str(i),
            "name": bus_name,
            "type": bus_type,
            "vm_pu": round(vm, 4),
            "theta_deg": round(float(np.degrees(theta)), 2),
        })

    grid_info = {
        "n_buses": len(list(grid.buses)),
        "n_generators": len(list(grid.generators)),
        "n_lines": len(list(grid.lines)),
        "n_loads": len(list(grid.loads)),
        "n_transformers": len(list(grid.transformers2w)),
    }

    op_hash = _hash_operating_point(np.asarray(pf_results.voltage, dtype=complex).real.reshape(-1))

    output = {
        "grid_name": grid_filename,
        "study": "EMT",
        "stable": results["stable"],
        "margin": results["margin"],
        "margin_definition": "rho_minus_one",
        "unstable_threshold": results["unstable_threshold"],
        "period": results["period"],
        "matches": 0,
        "op_hash": op_hash,
        "eigenvalues": eigenvalues_list,
        "critical_modes": critical_modes[:10],
        "bus_voltages": bus_voltages,
        "grid_info": grid_info,
        "n_eigenvalues": len(eigenvalues),
        "n_state_vars": problem.get_states_number() + problem.get_diff_var_number(),
        "elapsed_seconds": round(elapsed, 2),
    }

    output_dir = Path("precomputed_builds")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{Path(grid_filename).stem}_emt_{op_hash[:6]}.tensygrid"
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"Results saved to {output_path}")


def main() -> None:
    t_start = time.perf_counter()
    grid_filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRID
    grid = load_grid(grid_filename)
    results = run_emt_small_signal_analysis(grid)
    plot_results(results)
    save_results(results, grid, grid_filename, time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
