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
    for path in (repo_root / "src", repo_root / "trunk", repo_root):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    return repo_root


REPO_ROOT = ensure_repo_import_paths()

import VeraGridEngine.api as gce
from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Devices.Events.emt_events_group import EmtEventsGroup
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowOptions
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_simple_generator_emt_template
from VeraGridEngine.Templates.Emt.load_zip_emt_template import get_load_ZIP_emt_template
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.transformer_emt_template import get_transformer_emt_template
from VeraGridEngine.Utils.Symbolic.block import Var, find_name_in_block
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import (
    BranchImpedanceMode,
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtSolverTypes,
    SolverType,
    VarPowerFlowReferenceType,
)


def _set_constant_power_zip_coefficients(block: Any, model_name: str) -> None:
    block.set_parameter_in_model(var_name=f"a1_{model_name}", new_value=0.0)
    block.set_parameter_in_model(var_name=f"a2_{model_name}", new_value=0.0)
    block.set_parameter_in_model(var_name=f"a3_{model_name}", new_value=1.0)
    block.set_parameter_in_model(var_name=f"a4_{model_name}", new_value=0.0)
    block.set_parameter_in_model(var_name=f"a5_{model_name}", new_value=0.0)
    block.set_parameter_in_model(var_name=f"a6_{model_name}", new_value=1.0)


def _set_constant_impedance_reactive_zip_coefficients(block: Any, model_name: str) -> None:
    block.set_parameter_in_model(var_name=f"a1_{model_name}", new_value=0.0)
    block.set_parameter_in_model(var_name=f"a2_{model_name}", new_value=0.0)
    block.set_parameter_in_model(var_name=f"a3_{model_name}", new_value=0.0)
    block.set_parameter_in_model(var_name=f"a4_{model_name}", new_value=1.0)
    block.set_parameter_in_model(var_name=f"a5_{model_name}", new_value=0.0)
    block.set_parameter_in_model(var_name=f"a6_{model_name}", new_value=0.0)


def _set_event_constant_by_name(block: Any, var_factory: Any, var_name: str, value: float) -> None:
    for variable in list(block.event_dict.keys()) + list(block.parameters.keys()):
        if variable.name == var_name:
            block.event_dict[variable] = var_factory.add_const(value)
            return

    for variable in list(block.api_obj_mapping.values()):
        if variable is not None and variable.name == var_name:
            block.event_dict[variable] = var_factory.add_const(value)
            return

    raise KeyError(f"Variable '{var_name}' was not found in block '{block.name}'")


def _find_model_variable_by_name(block: Any, var_name: str) -> Var | None:
    variable = find_name_in_block(var_name, block)
    if variable is not None:
        return variable

    for mapped_variable in block.api_obj_mapping.values():
        if mapped_variable is not None and mapped_variable.name == var_name:
            return mapped_variable

    return None


def build_ieee9_grid() -> gce.MultiCircuit:
    grid_path = REPO_ROOT / "Grids_and_profiles" / "grids" / "IEEE 9 Bus.gridcal"
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    grid = gce.open_file(str(grid_path))
    grid.name = "IEEE 9 EMT example"
    return grid


def attach_emt_models(grid: gce.MultiCircuit) -> None:
    for bus in grid.buses:
        get_bus_emt_template(grid=grid, bus=bus)

    for idx, generator in enumerate(grid.generators, start=1):
        model = get_simple_generator_emt_template(vf=grid.var_factory, name=f"emt_gen_{idx}").block
        model.external_mapping[VarPowerFlowReferenceType.v_A] = model.in_vars[0]
        model.external_mapping[VarPowerFlowReferenceType.v_B] = model.in_vars[1]
        model.external_mapping[VarPowerFlowReferenceType.v_C] = model.in_vars[2]
        model.event_dict[model.in_vars[3]] = grid.var_factory.add_const(generator.P / grid.Sbase)
        model.event_dict[model.in_vars[4]] = grid.var_factory.add_const(1.0)
        set_emt_model(device=generator, model=model, var_factory=grid.var_factory)

    for transformer in grid.transformers2w:
        model = get_transformer_emt_template(vf=grid.var_factory, name=transformer.name).block
        set_emt_model(device=transformer, model=model, var_factory=grid.var_factory)

    for line in grid.lines:
        model = get_pi_line_emt_template(vf=grid.var_factory, phN=False, phA=True, phB=True, phC=True, name=line.name).block
        set_emt_model(device=line, model=model, var_factory=grid.var_factory)

    for shunt in grid.shunts:
        model = get_load_ZIP_emt_template(
            vf=grid.var_factory,
            phA=True,
            phB=True,
            phC=True,
            connection_type=None,
            name=f"emt_{shunt.name}",
        ).block
        model_name = model.name
        phase_q_pu = -shunt.B / grid.Sbase / 3.0
        _set_constant_impedance_reactive_zip_coefficients(block=model, model_name=model_name)
        _set_event_constant_by_name(
            block=model,
            var_factory=grid.var_factory,
            var_name=f"omega_{model_name}",
            value=2.0 * np.pi * grid.fBase,
        )
        for phase in ("A", "B", "C"):
            _set_event_constant_by_name(
                block=model,
                var_factory=grid.var_factory,
                var_name=f"P0_{phase}_{model_name}",
                value=0.0,
            )
            _set_event_constant_by_name(
                block=model,
                var_factory=grid.var_factory,
                var_name=f"Q0_{phase}_{model_name}",
                value=phase_q_pu,
            )
        set_emt_model(device=shunt, model=model, var_factory=grid.var_factory)

    for idx, load in enumerate(grid.loads, start=1):
        template_name = f"const_load_{idx}_{load.name}"
        model = get_load_ZIP_emt_template(
            vf=grid.var_factory,
            phA=True,
            phB=True,
            phC=True,
            connection_type=None,
            name=template_name,
        ).block
        _set_constant_power_zip_coefficients(block=model, model_name=model.name)
        set_emt_model(device=load, model=model, var_factory=grid.var_factory)


def add_first_load_step_event(grid: gce.MultiCircuit, event_time: float = 1.25e-4, scale: float = 1.2) -> None:
    if len(grid.loads) == 0:
        return

    load = grid.loads[0]
    model = load.emt_model
    event_group = EmtEventsGroup(name="load_step_event")
    grid.add_emt_events_group(event_group)

    phase_values = {
        "A": getattr(load, "P1", load.P / 3.0) / grid.Sbase,
        "B": getattr(load, "P2", load.P / 3.0) / grid.Sbase,
        "C": getattr(load, "P3", load.P / 3.0) / grid.Sbase,
    }
    for phase, p0 in phase_values.items():
        parameter = _find_model_variable_by_name(model, f"P0_{phase}_{model.name}")
        if parameter is None:
            raise KeyError(f"Could not find P0_{phase}_{model.name} for load event")
        grid.add_emt_event(
            EmtEvent(
                device=load,
                parameter=parameter,
                time=event_time,
                value=float(scale * p0),
                group=event_group,
            )
        )


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
        simulation_time=2.5e-4,
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
    grid = build_ieee9_grid()
    print(
        f"loaded IEEE9: buses={len(grid.buses)}, generators={len(grid.generators)}, "
        f"loads={len(grid.loads)}, lines={len(grid.lines)}, transformers={len(grid.transformers2w)}, shunts={len(grid.shunts)}",
        flush=True,
    )
    attach_emt_models(grid)
    print(f"attached EMT models in {perf_counter() - t0:.2f}s", flush=True)
    add_first_load_step_event(grid)

    t_pf = perf_counter()
    pf_driver = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf_driver.run()
    print(f"3ph power flow converged={pf_driver.results.converged} in {perf_counter() - t_pf:.2f}s", flush=True)
    if not bool(pf_driver.results.converged):
        raise RuntimeError("Three-phase IEEE 9 power flow did not converge")

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
    load_buses = [load.bus for load in problem.grid.loads[:3]]
    v_a_series = [y_arr[:, int(problem.get_var_idx(bus.emt_model.out_vars[0]))] for bus in load_buses]

    print("IEEE 9 EMT simulation")
    print(f"well_initialized={well_initialized}, converged={converged}, steps={len(t_arr) - 1}")
    print("time_s," + ",".join(f"vA_{bus.name}" for bus in load_buses))
    for idx in np.linspace(0, len(t_arr) - 1, min(8, len(t_arr)), dtype=int):
        values = ",".join(f"{series[idx]:.8e}" for series in v_a_series)
        print(f"{t_arr[idx]:.6e},{values}")


if __name__ == "__main__":
    main()
