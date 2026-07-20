# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from time import perf_counter
from typing import Any, cast

import numpy as np

from trunk.tensygrid.emt.ieee9_emt_simulation import (
    _set_constant_impedance_reactive_zip_coefficients,
    _set_constant_power_zip_coefficients,
    _set_event_constant_by_name,
    add_first_load_step_event,
    build_emt_options,
    build_ieee9_grid,
    build_power_flow_options,
    ensure_repo_import_paths,
)


ensure_repo_import_paths()

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Templates.Emt.load_zip_emt_multilinear_template import get_load_ZIP_emt_multilinear_template
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.simple_generator_emt_multilinear_template import get_simple_generator_emt_multilinear_template
from VeraGridEngine.Templates.Emt.transformer_emt_template import get_transformer_emt_template
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import VarPowerFlowReferenceType


def attach_multilinear_emt_models(grid: gce.MultiCircuit) -> None:
    for bus in grid.buses:
        get_bus_emt_template(grid=grid, bus=bus)

    for idx, generator in enumerate(grid.generators, start=1):
        model = get_simple_generator_emt_multilinear_template(vf=grid.var_factory, name=f"emt_gen_ml_{idx}").block
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
        model = get_load_ZIP_emt_multilinear_template(
            vf=grid.var_factory,
            phA=True,
            phB=True,
            phC=True,
            connection_type=None,
            name=f"emt_ml_{shunt.name}",
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
            _set_event_constant_by_name(block=model, var_factory=grid.var_factory, var_name=f"P0_{phase}_{model_name}", value=0.0)
            _set_event_constant_by_name(block=model, var_factory=grid.var_factory, var_name=f"Q0_{phase}_{model_name}", value=phase_q_pu)
        set_emt_model(device=shunt, model=model, var_factory=grid.var_factory)

    for idx, load in enumerate(grid.loads, start=1):
        template_name = f"const_load_ml_{idx}_{load.name}"
        model = get_load_ZIP_emt_multilinear_template(
            vf=grid.var_factory,
            phA=True,
            phB=True,
            phC=True,
            connection_type=None,
            name=template_name,
        ).block
        _set_constant_power_zip_coefficients(block=model, model_name=model.name)
        set_emt_model(device=load, model=model, var_factory=grid.var_factory)


def run_simulation() -> tuple[EmtProblemDae, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    t0 = perf_counter()
    grid = build_ieee9_grid()
    print(
        f"loaded IEEE9: buses={len(grid.buses)}, generators={len(grid.generators)}, "
        f"loads={len(grid.loads)}, lines={len(grid.lines)}, transformers={len(grid.transformers2w)}, shunts={len(grid.shunts)}",
        flush=True,
    )
    attach_multilinear_emt_models(grid)
    print(f"attached multilinear EMT models in {perf_counter() - t0:.2f}s", flush=True)
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
    print(f"built multilinear EMT problem in {perf_counter() - t_problem:.2f}s", flush=True)
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
    print(f"simulated multilinear EMT in {perf_counter() - t_sim:.2f}s", flush=True)
    return problem, t_arr, y_arr, dy_arr, bool(well_initialized), bool(converged)


def main() -> None:
    problem, t_arr, y_arr, _dy_arr, well_initialized, converged = run_simulation()
    load_buses = [load.bus for load in problem.grid.loads[:3]]
    v_a_series = [y_arr[:, int(problem.get_var_idx(bus.emt_model.out_vars[0]))] for bus in load_buses]

    print("IEEE 9 multilinear EMT simulation")
    print(f"well_initialized={well_initialized}, converged={converged}, steps={len(t_arr) - 1}")
    print("time_s," + ",".join(f"vA_{bus.name}" for bus in load_buses))
    for idx in np.linspace(0, len(t_arr) - 1, min(8, len(t_arr)), dtype=int):
        values = ",".join(f"{series[idx]:.8e}" for series in v_a_series)
        print(f"{t_arr[idx]:.6e},{values}")


if __name__ == "__main__":
    main()
