# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Simplified multilinear RMS small-signal analysis entry point."""

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
from VeraGridEngine.basic_structures import Logger
from VeraGridEngine.IO.veragrid.pack_unpack import parse_veragrid_data
from VeraGridEngine.IO.veragrid.zip_interface import get_frames_from_zip
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model


DEFAULT_GRID = "IEEE_9_Christoph.gridcal"


def load_grid(grid_filename: str = DEFAULT_GRID):
    """Load a VeraGrid grid from ``Grids_and_profiles/grids``."""
    grid_path = project_base / "Grids_and_profiles" / "grids" / grid_filename
    try:
        grid = vge.open_file(str(grid_path))
    except KeyError as ex:
        logger = Logger()
        data_dictionary, _, has_multiverse_data = get_frames_from_zip(file_name_zip=str(grid_path), logger=logger)
        if data_dictionary is None or has_multiverse_data or data_dictionary.get("symbolic_data") is None:
            raise
        print(f"Ignoring stale serialized symbolic data ({ex}); rebuilding RMS models from grid devices")
        data_dictionary["symbolic_data"] = {}
        grid = parse_veragrid_data(data=data_dictionary, logger=logger)

    print(f"Loaded grid: {grid_path}")
    return grid


def set_models(grid) -> None:
    """Attach phasor RMS models to buses, generators, branches, loads, and shunts."""
    def ensure_unique_device_names(devices, prefix: str) -> None:
        seen: dict[str, int] = {}
        for i, dev in enumerate(devices):
            base = str(dev.name).strip() if getattr(dev, "name", None) else f"{prefix}_{i}"
            if base not in seen:
                seen[base] = 0
                dev.name = base
            else:
                seen[base] += 1
                dev.name = f"{base}_{seen[base]}"

    ensure_unique_device_names(list(grid.generators), "gen")
    ensure_unique_device_names(list(grid.lines), "line")
    ensure_unique_device_names(list(grid.transformers2w), "trafo")
    ensure_unique_device_names(list(grid.loads), "load")
    ensure_unique_device_names(list(grid.shunts), "shunt")

    for bus in grid.buses:
        if bus.rms_model.empty():
            vge.initialize_bus_phasor_rms(bus, vf=grid.var_factory)

    for igen, gen in enumerate(grid.generators):
        if not gen.active or not gen.rms_model.empty():
            continue
        model = vge.get_complete_generator_template_phasor(grid.var_factory, name=f"Gen{igen}").block
        #model = vge.get_complete_generator_templatgete_phasor(grid.var_factory, name=f"Gen{igen}").block
        model = vge.to_implicit(model, grid.var_factory)
        set_rms_model(device=gen, model=model, var_factory=grid.var_factory)

    for line in grid.lines:
        if not line.active or not line.rms_model.empty():
            continue
        model = vge.get_line_phasor_rms_template(grid.var_factory, name=line.name).block
        model = vge.to_implicit(model, grid.var_factory)
        set_rms_model(device=line, model=model, var_factory=grid.var_factory)

    for load in grid.loads:
        if not load.active or not load.rms_model.empty():
            continue
        model = vge.get_load_phasor_current_rms_template(grid.var_factory, name=load.name).block
        set_rms_model(device=load, model=model, var_factory=grid.var_factory)

    for trafo in grid.transformers2w:
        if not trafo.active or not trafo.rms_model.empty():
            continue
        model = vge.initialize_trafo_rms(trafo, grid.var_factory, use_phasor_template=True).block
        model = vge.to_implicit(model, grid.var_factory)
        set_rms_model(device=trafo, model=model, var_factory=grid.var_factory)

    for shunt in grid.shunts:
        if not shunt.active or not shunt.rms_model.empty():
            continue
        model = vge.get_shunt_template(grid.var_factory, name=shunt.name, phasor=True).block
        model = vge.to_implicit(model, grid.var_factory)
        set_rms_model(device=shunt, model=model, var_factory=grid.var_factory)

    print("Attached RMS phasor models")


def run_small_signal_analysis(grid) -> dict:
    """Run power flow and small-signal analysis with ``RmsProblemMultilinear``."""
    pf_results = vge.power_flow(grid, vge.PowerFlowOptions(tolerance=1e-5))
    if not pf_results.converged:
        raise RuntimeError("Power flow did not converge")

    rms_options = vge.RmsOptions(
        time_step=0.01,
        simulation_time=1.0,
        tolerance=1e-6,
        max_iter=20,
        problem_type=vge.RmsProblemTypes.Multilinear,
    )
    problem = vge.RmsProblemMultilinear(grid=grid, options=rms_options, pf_results=pf_results)

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

    print(f"RmsProblemMultilinear states={problem.get_states_number()} diff_vars={problem.get_diff_var_number()}")
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
    """Plot and save the finite eigenvalues."""
    eigenvalues = results["eigenvalues"]
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "small_signal_analysis_main.png"

    fig, ax = plt.subplots(figsize=(9, 6))
    if len(eigenvalues):
        colors = np.real(eigenvalues) > 0.0
        ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), c=colors, cmap="coolwarm", s=45, alpha=0.85)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="k", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(f"RMS Multilinear Small-Signal Eigenvalues, margin={results['margin']:.3e}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {output_path}")


def _damping(re_: float, im: float) -> float:
    mag = np.sqrt(re_ * re_ + im * im)
    return -re_ / mag if mag > 1e-12 else 0.0


def _freq_hz(im: float) -> float:
    return abs(im) / (2 * np.pi)


def _hash_operating_point(x: np.ndarray) -> str:
    return hashlib.sha256(x.tobytes()).hexdigest()


def save_results(results: dict, grid, grid_filename: str, elapsed: float) -> None:
    eigenvalues = results["eigenvalues"]
    pf_results = results["pf_results"]
    problem = results["problem"]
    pf_matrix = results["participation_factors"]

    eigenvalues_list = [
        {"re": float(np.real(ev)), "im": float(np.imag(ev))}
        for ev in eigenvalues
    ]

    state_var_names = [
        str(v.name) if hasattr(v, "name") else f"state_{i}"
        for i, v in enumerate(problem.state_and_algebraic_vars)
    ]

    critical_modes = []
    for i, ev in enumerate(eigenvalues):
        re, im = float(np.real(ev)), float(np.imag(ev))
        if abs(im) <= 0.1:
            continue
        damping = float(_damping(re, im))

        participation = {}
        if pf_matrix is not None:
            pf_col = pf_matrix[:, i] if pf_matrix.ndim == 2 else pf_matrix
            for j, var_name in enumerate(state_var_names):
                if j < len(pf_col):
                    participation[str(var_name)] = float(pf_col[j])

        top_participation = dict(
            sorted(participation.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        )

        critical_modes.append(
            {
                "mode": i,
                "re": re,
                "im": im,
                "damping": damping,
                "freq_hz": float(_freq_hz(im)),
                "critical": bool(abs(damping) < 0.05),
                "participation_factors": top_participation,
            }
        )
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

        bus_voltages.append(
            {
                "id": bus.idtag if hasattr(bus, "idtag") else str(i),
                "name": bus_name,
                "type": bus_type,
                "vm_pu": round(vm, 4),
                "theta_deg": round(float(np.degrees(theta)), 2),
            }
        )

    grid_info = {
        "n_buses": len(list(grid.buses)),
        "n_generators": len(list(grid.generators)),
        "n_lines": len(list(grid.lines)),
        "n_loads": len(list(grid.loads)),
        "n_transformers": len(list(grid.transformers2w)),
    }

    x = problem.get_x0()
    op_hash = _hash_operating_point(x)

    output = {
        "grid_name": grid_filename,
        "stable": results["stable"],
        "margin": results["margin"],
        "matches": 0,
        "op_hash": op_hash,
        "eigenvalues": eigenvalues_list,
        "critical_modes": critical_modes[:10],
        "bus_voltages": bus_voltages,
        "grid_info": grid_info,
        "n_eigenvalues": len(eigenvalues),
        "n_state_vars": len(x),
        "elapsed_seconds": round(elapsed, 2),
    }

    output_dir = Path("precomputed_builds")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{Path(grid_filename).stem}_{op_hash[:6]}.tensygrid"
    with open(output_path, "wb") as f:
        pickle.dump(output, f)
    print(f"Results saved to {output_path}")


def main() -> None:
    t_start = time.perf_counter()
    grid_filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRID
    grid = load_grid(grid_filename)
    set_models(grid)
    results = run_small_signal_analysis(grid)
    plot_results(results)
    save_results(results, grid, grid_filename, time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
