# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
import sys

import numpy as np


def ensure_repo_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for path in (repo_root / "src", repo_root / "trunk", repo_root):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


ensure_repo_import_paths()

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowOptions
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Templates.Emt.load_RLC_emt_template import get_shunt_r_emt_template
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import get_generator_thevenin_rl_emt_template_with_ref
from VeraGridEngine.Utils.Symbolic.block import Var, find_name_in_block
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import (
    BranchImpedanceMode,
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtSolverTypes,
    ShuntConnectionType,
    SolverType,
)


def build_grid() -> gce.MultiCircuit:
    grid = gce.MultiCircuit(name="TensyGrid EMT two-bus RL example", Sbase=2.0, fbase=50.0)
    vnom = 10.0

    bus_source = gce.Bus(name="Source", Vnom=vnom, is_slack=True)
    bus_load = gce.Bus(name="Load", Vnom=vnom)
    grid.add_bus(bus_source)
    grid.add_bus(bus_load)

    line = gce.Line(name="Source_Load_Line", bus_from=bus_source, bus_to=bus_load, length=10.0, rate=900.0)
    tower = gce.OverheadLineType(name="EMT_Demo_Tower", Vnom=vnom)
    wire = gce.Wire(
        name="Panther 30/7 ACSR",
        diameter=21.0,
        diameter_internal=9.0,
        is_tube=True,
        r=0.1363,
        max_current=1.0,
    )
    tower.add_wire_relationship(wire=wire, xpos=-12.65, ypos=27.5, phase=1)
    tower.add_wire_relationship(wire=wire, xpos=0.0, ypos=27.5, phase=2)
    tower.add_wire_relationship(wire=wire, xpos=12.65, ypos=27.5, phase=3)
    tower.compute()
    line.apply_template(tower, grid.Sbase, grid.fBase)

    r_phase_ohm = 100.0
    v_phase_kv = vnom / np.sqrt(3.0)
    p_phase_mw = v_phase_kv * v_phase_kv / r_phase_ohm
    load = gce.Load(
        name="Grounded_R_Load",
        P=3.0 * p_phase_mw,
        Q=0.0,
        P1=p_phase_mw,
        P2=p_phase_mw,
        P3=p_phase_mw,
        Q1=0.0,
        Q2=0.0,
        Q3=0.0,
    )
    load.conn = ShuntConnectionType.GroundedStar

    generator = gce.Generator(name="Thevenin_Source", vset=1.0, Snom=grid.Sbase, freq=50.0, r1=0.001, x1=1.7)

    grid.add_line(line)
    grid.add_generator(bus=bus_source, api_obj=generator)
    grid.add_load(bus=bus_load, api_obj=load)

    for bus in grid.buses:
        get_bus_emt_template(grid=grid, bus=bus)

    set_emt_model(
        device=generator,
        model=get_generator_thevenin_rl_emt_template_with_ref(vf=grid.var_factory).block,
        var_factory=grid.var_factory,
    )
    set_emt_model(
        device=line,
        model=get_pi_line_emt_template(vf=grid.var_factory, phN=False, phA=True, phB=True, phC=True).block,
        var_factory=grid.var_factory,
    )
    set_emt_model(
        device=load,
        model=get_shunt_r_emt_template(vf=grid.var_factory, phA=True, phB=True, phC=True).block,
        var_factory=grid.var_factory,
    )

    return grid


def build_power_flow_options() -> PowerFlowOptions:
    return PowerFlowOptions(
        solver_type=SolverType.NR,
        retry_with_other_methods=False,
        verbose=0,
        initialize_with_existing_solution=True,
        tolerance=1.0e-6,
        max_iter=25,
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
        time_step=5.0e-6,
        simulation_time=1.0e-3,
        tolerance=1.0e-6,
        solver_type=EmtSolverTypes.Symbolic,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.PseudoTransient,
        verbose=0,
    )


def get_series(problem: EmtProblemDae, y_arr: np.ndarray, variable_name: str) -> np.ndarray:
    variable: Var | None = find_name_in_block(variable_name, problem.sys_block)
    if variable is None:
        raise KeyError(f"Variable '{variable_name}' was not found in the EMT problem")
    return y_arr[:, int(problem.get_var_idx(variable))]


def run_simulation() -> tuple[EmtProblemDae, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    grid = build_grid()
    pf_driver = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf_driver.run()

    if not bool(pf_driver.results.converged):
        raise RuntimeError("Three-phase power flow did not converge; EMT initialization cannot be seeded")

    options = build_emt_options()
    problem = EmtProblemDae(grid=grid, options=options, pf_results_3ph=pf_driver.results, pf_results=None)
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=float(options.simulation_time),
        h=float(options.time_step),
        method=options.integration_method,
    )
    t_arr, y_arr, dy_arr, well_initialized, converged = solver.simulate(boundary_updater=cast(Any, problem))
    return problem, t_arr, y_arr, dy_arr, bool(well_initialized), bool(converged)


def main() -> None:
    problem, t_arr, y_arr, _dy_arr, well_initialized, converged = run_simulation()
    load_bus_va = get_series(problem, y_arr, "v_A_Load_emt_template")
    load_bus_vb = get_series(problem, y_arr, "v_B_Load_emt_template")
    load_bus_vc = get_series(problem, y_arr, "v_C_Load_emt_template")

    print("EMT two-bus RL simulation")
    print(f"well_initialized={well_initialized}, converged={converged}, steps={len(t_arr) - 1}")
    print("time_s,vA_load,vB_load,vC_load")
    for idx in np.linspace(0, len(t_arr) - 1, 8, dtype=int):
        print(f"{t_arr[idx]:.6e},{load_bus_va[idx]:.8e},{load_bus_vb[idx]:.8e},{load_bus_vc[idx]:.8e}")


if __name__ == "__main__":
    main()
