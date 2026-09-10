"""IEEE9 EMT test using the engine GFM template, not validation models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt.emt_ieee9 import (
    attach_emt_models,
    build_emt_options,
    build_ieee9_grid,
    build_power_flow_options,
)
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Templates.Emt.emt_gfm_converter import build_emt_gfm_aggregated_model
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model


TIME_STEP = 5.0e-6
SIMULATION_TIME = 1.0e-3


def _set_event(block: Block, name: str, value: float, vf) -> None:
    matches = [
        variable
        for owner in block.get_all_blocks()
        for variable in owner.event_dict
        if variable.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one GFM event parameter {name!r}, found {len(matches)}")
    variable = matches[0]
    for owner in block.get_all_blocks():
        if variable in owner.event_dict:
            owner.event_dict[variable] = vf.add_const(float(value))


def _signal(problem: EmtProblemDae, values: np.ndarray, block: Block, name: str):
    variable = next((item for item in block.get_all_vars() if item.name == name), None)
    if variable is None or variable.uid not in problem.uid2idx_vars:
        return None
    return values[:, int(problem.get_var_idx(variable))]


def main() -> None:
    grid = build_ieee9_grid()
    pf = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError("IEEE9 power flow failed")

    attach_emt_models(grid)
    generator = sorted(
        (item for item in grid.generators if not item.bus.is_slack),
        key=lambda item: item.bus.name,
    )[0]
    bus_index = grid.buses.index(generator.bus)
    p_ref = float(np.real(pf.results.Sbus[bus_index]) / grid.Sbase)
    q_ref = float(np.imag(pf.results.Sbus[bus_index]) / grid.Sbase)
    v_ref = float(np.sqrt(2.0) * abs(pf.results.voltage[bus_index]))
    model = build_emt_gfm_aggregated_model(
        grid.var_factory, name=f"engine_gfm_{generator.bus.name}"
    )
    _set_event(model, "P_ref", p_ref, grid.var_factory)
    _set_event(model, "Q_ref", q_ref, grid.var_factory)
    _set_event(model, "V_ref", v_ref, grid.var_factory)
    set_emt_model(generator, model, grid.var_factory)

    options = build_emt_options()
    options.time_step = TIME_STEP
    options.simulation_time = SIMULATION_TIME
    problem = EmtProblemDae(
        grid=grid, options=options, pf_results=pf.results, pf_results_3ph=pf3.results
    )
    solver = build_emt_solver(
        options=options, problem=problem, t0=0.0, t_end=SIMULATION_TIME,
        h=TIME_STEP, method=options.integration_method,
    )
    time, values, _, initialized, converged = solver.simulate(
        boundary_updater=cast(Any, problem)
    )
    time, values = np.asarray(time), np.asarray(values)
    print(
        f"builder=build_emt_gfm_aggregated_model bus={generator.bus.name} "
        f"initialized={initialized} converged={converged} steps={len(time)-1}"
    )
    traces = {name: _signal(problem, values, generator.emt_model, name)
              for name in ("omega", "P", "Q")}
    for name, trace in traces.items():
        if trace is not None:
            print(f"{name}={trace[0]:.9g}->{trace[-1]:.9g} delta={trace[-1]-trace[0]:+.6e}")

    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, ("omega", "P", "Q")):
        trace = traces[name]
        if trace is not None:
            axis.plot(1e3 * time, trace)
        axis.set_ylabel(name)
        axis.grid(alpha=0.3)
    axes[-1].set_xlabel("Time [ms]")
    axes[0].set_title("IEEE9 GFM using VeraGridEngine template")
    output = Path(__file__).with_name("ieee9_emt_gfm_engine_template.png")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"plot={output}")


if __name__ == "__main__":
    main()
