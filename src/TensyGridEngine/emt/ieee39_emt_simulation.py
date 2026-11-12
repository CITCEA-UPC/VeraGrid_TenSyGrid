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


def configure_ieee39_simple_machines(grid: gce.MultiCircuit) -> None:
    """Parameterize the demo's reduced machines from local IEEE39 GENROU data.

    Records are the native-base data in VeraGrid's
    benchmarks/dynamics_andes/small_signal/IEEE39/ieee39_dynamic_model.py.
    This is a single-field reduction, without GENROU damper windings or its
    excitation/governor models. Xd, Xq, Xl, Xd' and Tdo' retain their meanings.
    Missing zero-sequence data uses leakage reactance and armature resistance
    as explicit demo assumptions. Static network/PF impedances are unchanged.
    """
    # Sn [MVA], H [s], Ra, Xl, Xd', Xd, Xq [machine pu at 60 Hz], Tdo' [s].
    records = (
        (1040., 8.4, .0014, .125, .31, 1., .69, 10.2),
        (836., 6.06, .027, .35, .697, 2.95, 2.82, 6.56),
        (843.7, 7.16, .00386, .304, .531, 2.495, 2.37, 5.7),
        (1174.8, 5.72, .00222, .295, .436, 2.62, 2.58, 5.69),
        (1080.2, 5.2, .0014, .54, 1.32, 6.7, 6.2, 5.4),
        (1085.7, 6.96, .0615, .224, .5, 2.54, 2.41, 7.3),
        (1025.2, 5.28, .00268, .322, .49, 2.95, 2.92, 5.66),
        (970.2, 4.86, .00686, .28, .57, 2.9, 2.8, 6.7),
        (1684.1, 6.9, .003, .298, .57, 2.106, 2.05, 4.79),
        (1199., 100., .001, .03, .06, .2, .19, 7.),
    )
    bus_positions = {bus.idtag: index for index, bus in enumerate(grid.buses)}
    machines = sorted(grid.generators, key=lambda gen: bus_positions[gen.bus.idtag])
    if len(machines) != len(records):
        raise ValueError("IEEE39 machine configuration requires ten generators")
    for generator, (rating, inertia, ra, xl, xd1, xd, xq, td0) in zip(machines, records):
        model = generator.emt_model
        if model is None:
            raise ValueError("Attach simple EMT models before configuring IEEE39 machines")
        variables = {var.name: var for var in list(model.parameters) + list(model.event_dict)}
        impedance_scale = grid.Sbase / rating
        reactance_scale = impedance_scale * grid.fBase / 60.0
        leakage, direct, transient = xl * reactance_scale, xd * reactance_scale, xd1 * reactance_scale
        mutual = direct - leakage
        field_leakage = (transient - leakage) * mutual / (direct - transient)
        values = {
            'Ra': ra * impedance_scale, 'Ld': direct, 'La': leakage,
            'Lmq': (xq - xl) * reactance_scale, 'Lf': field_leakage,
            'Rf': (mutual + field_leakage) / (2 * np.pi * grid.fBase * td0),
            'H': inertia * rating / grid.Sbase,
            'L0': leakage, 'R0': ra * impedance_scale,
        }
        configured_uids = {variables[name].uid for name in values}
        for reference, variable in list(model.api_obj_mapping.items()):
            if variable.uid in configured_uids:
                del model.api_obj_mapping[reference]
        for name, value in values.items():
            variable = variables[name]
            destination = model.parameters if variable in model.parameters else model.event_dict
            destination[variable] = grid.var_factory.add_const(float(value))


def configure_ieee39_stability_profile(
        grid: gce.MultiCircuit, voltage: np.ndarray, impedance_fraction: float = 0.25,
        damping_rate: float = 0.25, excitation_gain: float = 2.0,
) -> None:
    """Apply an experimental ZIP mix, speed damping and static voltage regulator.

    Call after simple-machine configuration and balanced PF, before building
    the EMT problem. ZIP coefficients retain the PF P/Q at each actual bus
    voltage. D/(2H) is in 1/s. Excitation adds
    vf_nominal * gain * (1 - Vrms/Vref), with no exciter delay or limits.
    This demonstration profile is not the original IEEE39 GENROU model.
    """
    values = (impedance_fraction, damping_rate, excitation_gain)
    if (not np.all(np.isfinite(values)) or not 0 <= impedance_fraction <= 1
            or damping_rate < 0 or excitation_gain < 0):
        raise ValueError("Use a ZIP fraction in [0, 1] and nonnegative finite gains")
    magnitudes = np.abs(np.asarray(voltage))
    if magnitudes.shape != (len(grid.buses),) or np.any(magnitudes <= 0):
        raise ValueError("Supply positive balanced PF voltages in grid bus order")
    if not np.all(np.isfinite(magnitudes)):
        raise ValueError("PF voltages must be finite")
    bus_index = {bus.idtag: i for i, bus in enumerate(grid.buses)}

    def entries(model):
        return {var.name: (var, table) for block in model.get_all_blocks()
                for table in (block.event_dict, block.parameters) for var in table}

    def set_value(entry, value):
        var, table = entry
        table[var] = grid.var_factory.add_const(float(value))

    for load in grid.loads:
        parameters = entries(load.emt_model)
        eps = float(parameters['eps'][1][parameters['eps'][0]].value)
        v0 = float(parameters['V0'][1][parameters['V0'][0]].value)
        peak = np.sqrt(2 * magnitudes[bus_index[load.bus.idtag]] ** 2 + eps)
        ratio = (peak + eps) / v0
        for name, value in {'a1': impedance_fraction / ratio**2, 'a2': 0.,
                            'a3': 1-impedance_fraction,
                            'a4': impedance_fraction / ratio**2, 'a5': 0.,
                            'a6': 1-impedance_fraction}.items():
            set_value(parameters[name], value)

    for generator in grid.generators:
        model = generator.emt_model
        parameters = entries(model)
        inertia = float(parameters['H'][1][parameters['H'][0]].value)
        set_value(parameters['D'], 2 * inertia * damping_rate)
        if excitation_gain:
            import VeraGridEngine.Utils.Symbolic.symbolic as sym

            nominal = float(magnitudes[bus_index[generator.bus.idtag]])
            terminal_rms = sym.sqrt(sum(v * v for v in model.in_vars[:3]) / 3)
            omega_base = parameters['omega_base'][0]
            row = next(i for i, var in enumerate(model.state_vars) if var.name == 'psi_f')
            model.state_eqs[row] = (model.state_eqs[row]
                + omega_base * model.in_vars[4] * excitation_gain * (1-terminal_rms/nominal))


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
    configure_ieee39_simple_machines(grid)
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
