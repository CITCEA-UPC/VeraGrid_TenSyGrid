# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, cast
import sys

import numpy as np


def ensure_repo_import_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    for path in (repo_root / "src", repo_root / "trunk", repo_root, Path(__file__).resolve().parent):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    return repo_root


REPO_ROOT = ensure_repo_import_paths()

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowOptions
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Utils.Symbolic.block import Var, find_name_in_block
from VeraGridEngine.enumerations import (
    BranchImpedanceMode,
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtSolverTypes,
    SolverType,
)

from ieee9_emt_simulation import add_first_load_step_event, attach_emt_models


def build_ieee39_grid() -> gce.MultiCircuit:
    grid_path = REPO_ROOT / "Grids_and_profiles" / "grids" / "IEEE39.gridcal"
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    grid = gce.open_file(str(grid_path))
    grid.name = "IEEE 39 EMT example"
    return grid


def build_power_flow_options() -> PowerFlowOptions:
    return PowerFlowOptions(
        solver_type=SolverType.NR,
        retry_with_other_methods=False,
        verbose=0,
        initialize_with_existing_solution=True,
        tolerance=1.0e-6,
        max_iter=30,
        control_q=False,
        control_taps_modules=True,
        control_taps_phase=True,
        control_remote_voltage=True,
        orthogonalize_controls=True,
        apply_temperature_correction=True,
        branch_impedance_tolerance_mode=BranchImpedanceMode.Specified,
        distributed_slack=False,
        ignore_single_node_islands=False,
        trust_radius=1.0,
        backtracking_parameter=0.05,
        use_stored_guess=False,
        initialize_angles=False,
        generate_report=False,
    )


def build_emt_options() -> EmtOptions:
    return EmtOptions(
        time_step=2.5e-5,
        simulation_time=5.0e-5,
        tolerance=1.0e-6,
        solver_type=EmtSolverTypes.StructuralCompiled,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        verbose=0,
    )


def get_series(problem: EmtProblemDae, y_arr: np.ndarray, variable_name: str) -> np.ndarray:
    variable: Var | None = find_name_in_block(variable_name, problem.sys_block)
    if variable is None:
        raise KeyError(f"Variable '{variable_name}' was not found in the EMT problem")
    return y_arr[:, int(problem.get_var_idx(variable))]


def run_simulation() -> tuple[EmtProblemDae, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    t0 = perf_counter()
    grid = build_ieee39_grid()
    print(
        f"loaded IEEE39: buses={len(grid.buses)}, generators={len(grid.generators)}, "
        f"loads={len(grid.loads)}, lines={len(grid.lines)}, transformers={len(grid.transformers2w)}, shunts={len(grid.shunts)}",
        flush=True,
    )
    attach_emt_models(grid)
    print(f"attached EMT models in {perf_counter() - t0:.2f}s", flush=True)
    add_first_load_step_event(grid, event_time=2.5e-5, scale=1.2)

    t_pf = perf_counter()
    pf_driver = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf_driver.run()
    print(f"3ph power flow converged={pf_driver.results.converged} in {perf_counter() - t_pf:.2f}s", flush=True)
    if not bool(pf_driver.results.converged):
        raise RuntimeError("Three-phase IEEE 39 power flow did not converge")

    options = build_emt_options()
    t_problem = perf_counter()
    problem = EmtProblemDae(grid=grid, options=options, pf_results_3ph=pf_driver.results, pf_results=None)
    print(f"built EMT problem in {perf_counter() - t_problem:.2f}s", flush=True)
    t_solver = perf_counter()
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=float(options.simulation_time),
        h=float(options.time_step),
        method=options.integration_method,
    )
    print(f"built EMT solver in {perf_counter() - t_solver:.2f}s", flush=True)
    t_sim = perf_counter()
    t_arr, y_arr, dy_arr, well_initialized, converged = solver.simulate(boundary_updater=cast(Any, problem))
    print(f"simulated EMT in {perf_counter() - t_sim:.2f}s", flush=True)
    return problem, t_arr, y_arr, dy_arr, bool(well_initialized), bool(converged)


def main() -> None:
    problem, t_arr, y_arr, _dy_arr, well_initialized, converged = run_simulation()
    first_load_bus = problem.grid.loads[0].bus
    v_a_idx = int(problem.get_var_idx(first_load_bus.emt_model.out_vars[0]))
    v_a = y_arr[:, v_a_idx]

    print("IEEE 39 EMT simulation")
    print(f"well_initialized={well_initialized}, converged={converged}, steps={len(t_arr) - 1}")
    print(f"time_s,vA_{first_load_bus.name}")
    for idx in np.linspace(0, len(t_arr) - 1, min(4, len(t_arr)), dtype=int):
        print(f"{t_arr[idx]:.6e},{v_a[idx]:.8e}")


if __name__ == "__main__":
    main()
