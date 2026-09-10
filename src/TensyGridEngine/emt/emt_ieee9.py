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
from VeraGridEngine.Simulations.EMT.emt_problem_factory import build_emt_problem
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver, PowerFlowOptions
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Templates.Emt.load_RLC_emt_template import get_shunt_rlc_combo_emt_template
from VeraGridEngine.Templates.Emt.generator_sauer_pai_emt_multilinear_template import (
    get_complete_generator_template_emt_multilinear,
)
from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_complete_generator_template_emt
from VeraGridEngine.Templates.Emt.load_zip_emt_template import get_load_ZIP_emt_template
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.transformer_emt_template import get_series_transformer_emt_template
from VeraGridEngine.Utils.Symbolic.block import Var, find_name_in_block
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import (
    BranchImpedanceMode,
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtProblemTypes,
    EmtSolverTypes,
    SolverType,
    ShuntConnectionType,
    VarPowerFlowReferenceType,
)


def _set_constant_power_zip_coefficients(block: Any, model_name: str) -> None:
    for coefficient, value in (("a1", 0.0), ("a2", 0.0), ("a3", 1.0),
                               ("a4", 0.0), ("a5", 0.0), ("a6", 1.0)):
        _set_model_parameter(block, coefficient, model_name, value)


def _set_constant_impedance_reactive_zip_coefficients(block: Any, model_name: str) -> None:
    for coefficient, value in (("a1", 0.0), ("a2", 0.0), ("a3", 0.0),
                               ("a4", 1.0), ("a5", 0.0), ("a6", 0.0)):
        _set_model_parameter(block, coefficient, model_name, value)


def _set_model_parameter(block: Any, base_name: str, model_name: str, value: float) -> None:
    """Set a template parameter using either current or legacy suffixed naming."""
    available_names = {
        variable.name
        for variable in list(block.event_dict.keys()) + list(block.parameters.keys())
    }
    for candidate in (base_name, f"{base_name}_{model_name}"):
        if candidate in available_names:
            block.set_parameter_in_model(var_name=candidate, new_value=value)
            return
    raise KeyError(f"Parameter '{base_name}' was not found in block '{block.name}'")


def _set_event_constant_by_name(block: Any, var_factory: Any, var_name: str, value: float) -> None:
    suffix = f"_{block.name}"
    candidate_names = {var_name}
    if var_name.endswith(suffix):
        candidate_names.add(var_name[:-len(suffix)])

    for variable in list(block.event_dict.keys()) + list(block.parameters.keys()):
        if variable.name in candidate_names:
            block.event_dict[variable] = var_factory.add_const(value)
            return

    for variable in list(block.api_obj_mapping.values()):
        if variable is not None and variable.name in candidate_names:
            block.event_dict[variable] = var_factory.add_const(value)
            return

    raise KeyError(f"Variable '{var_name}' was not found in block '{block.name}'")


def _find_model_variable_by_name(block: Any, var_name: str) -> Var | None:
    suffix = f"_{block.name}"
    candidate_names = [var_name]
    if var_name.endswith(suffix):
        candidate_names.append(var_name[:-len(suffix)])

    for candidate in candidate_names:
        variable = find_name_in_block(candidate, block)
        if variable is not None:
            return variable

    for mapped_variable in block.api_obj_mapping.values():
        if mapped_variable is not None and mapped_variable.name in candidate_names:
            return mapped_variable

    return None


def build_ieee9_grid() -> gce.MultiCircuit:
    grid_path = REPO_ROOT / "Grids_and_profiles" / "grids" / "IEEE 9 Bus.gridcal"
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    grid = gce.open_file(str(grid_path))
    grid.name = "IEEE 9 EMT example"

    # This legacy archive's generator.csv does not contain the sequence fields,
    # so Generator falls back to near-zero placeholder impedances. EMT machine
    # reactance belongs here on the generators; all nine network branches remain
    # lines exactly as stored in the MultiCircuit.
    for generator in grid.generators:
        generator.Snom = grid.Sbase
        generator.R1 = 0.002 if generator.bus.is_slack else 0.001
        generator.X1 = 0.25 if generator.bus.is_slack else 1.7

    return grid


def attach_emt_models(
    grid: gce.MultiCircuit,
    frozen_controls: bool = False,
    multilinear_controls: bool = True,
    multilinear_machine: bool = True,
) -> None:
    for bus in grid.buses:
        get_bus_emt_template(grid=grid, bus=bus)

    for idx, generator in enumerate(grid.generators, start=1):
        generator_factory = (
            get_complete_generator_template_emt_multilinear
            if multilinear_machine else get_complete_generator_template_emt
        )
        model = generator_factory(
            vf=grid.var_factory,
            name=f"emt_complete_generator_{idx}",
            conventional_three_phase_base=True,
            frozen_controls=frozen_controls,
            multilinear_controls=multilinear_controls,
        ).block
        set_emt_model(device=generator, model=model, var_factory=grid.var_factory)
        # K is inverse droop. K=40 corresponds to 2.5% droop and stabilizes the
        # original non-lifted IEEE9 electromechanical mode while retaining zero
        # artificial mechanical damping.
        if not frozen_controls:
            model.set_parameter_in_model(var_name="K", new_value=40.0)

    for transformer in grid.transformers2w:
        model = get_series_transformer_emt_template(
            vf=grid.var_factory,
            name=transformer.name,
            r=transformer.R,
            x=transformer.X,
            tap_module=transformer.tap_module,
        ).block
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
            conventional_three_phase_base=True,
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
        model = get_shunt_rlc_combo_emt_template(
            vf=grid.var_factory,
            include_r=abs(float(load.P)) > 1.0e-15,
            include_l=abs(float(load.Q)) > 1.0e-15,
            include_c=False,
            phA=True,
            phB=True,
            phC=True,
            connection_type=ShuntConnectionType.FloatingStar,
            name=f"rlc_load_{idx}_{load.name}",
        ).block
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
        simulation_time=1.0e-2,
        tolerance=1.0e-6,
        solver_type=EmtSolverTypes.StructuralCompiled,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        problem_type=EmtProblemTypes.Multilinear,
        conventional_three_phase_base=True,
        verbose=0,
    )


def get_series(problem: EmtProblemDae, y_arr: np.ndarray, variable_name: str) -> np.ndarray:
    variable: Var | None = find_name_in_block(variable_name, problem.sys_block)
    if variable is None:
        raise KeyError(f"Variable '{variable_name}' was not found in the EMT problem")
    return y_arr[:, int(problem.get_var_idx(variable))]


def audit_periodic_initialization(
    problem: EmtProblemDae,
    y_arr: np.ndarray,
    dy_arr: np.ndarray,
    tolerance: float = 1.0e-5,
) -> float:
    """Validate stationary machine states and rotating ABC device states at t=0."""
    differential_by_state_uid = {
        differential.base_var.uid: differential
        for differential in problem.get_diff_vars()
        if differential.base_var is not None
    }
    largest_residual = 0.0

    stationary_names = {
        "omega_", "psi_d_", "psi_q_", "psi_0_", "e_qp_", "e_dp_",
        "psi_pp_d_", "psi_pp_q_", "y_gov0", "y2_3_gov", "Vf",
        "y_exciter1", "y_exciter2", "y_exciter3", "y_exciter4",
    }
    for generator in problem.grid.generators:
        for variable_name in stationary_names:
            variable = find_name_in_block(variable_name, generator.emt_model)
            if variable is None or variable.uid not in differential_by_state_uid:
                continue
            differential = differential_by_state_uid[variable.uid]
            residual = abs(float(dy_arr[0, problem.get_diff_var_idx(differential)]))
            largest_residual = max(largest_residual, residual)

    omega_base = 2.0 * np.pi * problem.grid.fBase
    device_groups = (
        (problem.grid.lines, ("i_ser_{name}_A", "i_ser_{name}_B", "i_ser_{name}_C")),
        (problem.grid.transformers2w, ("i_ser_A", "i_ser_B", "i_ser_C")),
        (problem.grid.loads, ("iL_A", "iL_B", "iL_C")),
    )
    for devices, phase_name_patterns in device_groups:
        for device in devices:
            phase_names = tuple(pattern.format(name=device.name) for pattern in phase_name_patterns)
            phase_variables = [find_name_in_block(name, device.emt_model) for name in phase_names]
            if any(variable is None for variable in phase_variables):
                continue
            differentials = [
                differential_by_state_uid.get(variable.uid) for variable in phase_variables
            ]
            if any(differential is None for differential in differentials):
                continue
            values = np.asarray([
                y_arr[0, problem.get_var_idx(variable)] for variable in phase_variables
            ])
            derivatives = np.asarray([
                dy_arr[0, problem.get_diff_var_idx(differential)]
                for differential in differentials
            ])
            expected = (omega_base / np.sqrt(3.0)) * np.asarray([
                values[2] - values[1],
                values[0] - values[2],
                values[1] - values[0],
            ])
            largest_residual = max(
                largest_residual,
                float(np.max(np.abs(derivatives - expected))),
            )

    if largest_residual > tolerance:
        raise RuntimeError(
            "IEEE9 EMT initialization is not periodic steady state: "
            f"largest derivative residual={largest_residual:.6e}, tolerance={tolerance:.6e}"
        )
    return largest_residual


def audit_exciter_voltage_references(problem: EmtProblemDae, tolerance: float = 1.0e-10) -> float:
    """Verify every AVR reference against its complete equilibrium input balance."""
    x0 = problem.get_x0()
    largest_mismatch = 0.0
    usref_parameters = [
        parameter for parameter in problem.get_variable_parameters()
        if parameter.name == "UsRefPu"
    ]
    if len(usref_parameters) != len(problem.grid.generators):
        raise RuntimeError(
            f"Expected one UsRefPu per generator; found {len(usref_parameters)} "
            f"for {len(problem.grid.generators)} generators"
        )
    print("generator,Vm,y1,Vpss,y2,y3,UsRefPu,required_UsRefPu,mismatch,error_minus_y3")
    for generator_index, generator in enumerate(problem.grid.generators):
        block = generator.emt_model
        variables = {
            name: find_name_in_block(name, block)
            for name in ("Vm", "y_exciter1", "V_pss", "y_exciter2", "y_exciter3")
        }
        missing = [name for name, variable in variables.items() if variable is None]
        if missing:
            raise RuntimeError(f"{generator.name}: missing AVR variables {missing}")

        def initial_value(name: str) -> float:
            variable = variables[name]
            assert variable is not None
            return float(x0[int(problem.get_var_idx(variable))])

        vm = initial_value("Vm")
        y1 = initial_value("y_exciter1")
        vpss = initial_value("V_pss")
        y2 = initial_value("y_exciter2")
        y3 = initial_value("y_exciter3")
        usref_variable = usref_parameters[generator_index]
        runtime_index = problem.uid2idx_event_params.get(usref_variable.uid)
        if runtime_index is None:
            raise RuntimeError(f"{generator.name}: UsRefPu is absent from the runtime parameter vector")
        usref = float(problem.event_params_values[runtime_index])

        # At equilibrium: exciter_error = UsRefPu + Vpss - y1 - y2 = y3.
        required_usref = y1 + y2 - vpss + y3
        mismatch = usref - required_usref
        error_minus_y3 = usref + vpss - y1 - y2 - y3
        largest_mismatch = max(largest_mismatch, abs(mismatch), abs(error_minus_y3))
        print(
            f"{generator.name},{vm:.12e},{y1:.12e},{vpss:.12e},{y2:.12e},{y3:.12e},"
            f"{usref:.12e},{required_usref:.12e},{mismatch:+.3e},{error_minus_y3:+.3e}"
        )

    if largest_mismatch > tolerance:
        raise RuntimeError(
            "IEEE9 AVR reference initialization is inconsistent: "
            f"largest mismatch={largest_mismatch:.6e}, tolerance={tolerance:.6e}"
        )
    return largest_mismatch


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
    # Events can be added after the periodic operating point has been verified.

    t_pf = perf_counter()
    balanced_pf_driver = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    balanced_pf_driver.run()
    if not bool(balanced_pf_driver.results.converged):
        raise RuntimeError("Balanced IEEE 9 power flow did not converge")

    pf_driver = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf_driver.run()
    print(f"3ph power flow converged={pf_driver.results.converged} in {perf_counter() - t_pf:.2f}s", flush=True)
    if not bool(pf_driver.results.converged):
        raise RuntimeError("Three-phase IEEE 9 power flow did not converge")

    options = build_emt_options()
    t_problem = perf_counter()
    problem = build_emt_problem(
        grid=grid,
        options=options,
        pf_results_3ph=pf_driver.results,
        pf_results=balanced_pf_driver.results,
    )
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
    problem, t_arr, y_arr, dy_arr, well_initialized, converged = run_simulation()
    exciter_reference_mismatch = audit_exciter_voltage_references(problem)
    periodic_residual = audit_periodic_initialization(problem, y_arr, dy_arr)
    load_buses = [load.bus for load in problem.grid.loads[:3]]
    v_a_series = [y_arr[:, int(problem.get_var_idx(bus.emt_model.out_vars[0]))] for bus in load_buses]

    print("IEEE 9 EMT simulation")
    print(f"well_initialized={well_initialized}, converged={converged}, steps={len(t_arr) - 1}")
    print(f"largest_periodic_initialization_residual={periodic_residual:.8e}")
    print(f"largest_exciter_reference_mismatch={exciter_reference_mismatch:.8e}")
    print("time_s," + ",".join(f"vA_{bus.name}" for bus in load_buses))
    for idx in np.linspace(0, len(t_arr) - 1, min(8, len(t_arr)), dtype=int):
        values = ",".join(f"{series[idx]:.8e}" for series in v_a_series)
        print(f"{t_arr[idx]:.6e},{values}")


if __name__ == "__main__":
    main()
