"""Shared IEEE9 construction used by the EMT comparison demos."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from TensyGridEngine.emt import ieee9_emt_ibr_simulation as ieee9
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.EMT.problems.emt_problem_multilinear import (
    EmtProblemMultilinear,
)
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph


GOVERNOR_K = float(os.environ.get("IEEE9_DEMO_GOVERNOR_K", "40.0"))


@dataclass
class Ieee9Case:
    """Objects needed to simulate or linearize one IEEE9 formulation."""

    grid: object
    problem: EmtProblemDae
    power_flow: object
    gfm_model: object


def build_ieee9_case(
    multilinear: bool,
    time_step: float,
    simulation_time: float,
    integration_method=None,
    initialization_method=None,
) -> Ieee9Case:
    """Build IEEE9 with one GFM and return the requested EMT problem class."""
    grid, _gfl, _gfl_buses, gfm_buses, gfm_blocks, validation = ieee9.build_ibr_grid(
        n_gfl=0,
        n_gfm=1,
        multilinear_inverters=multilinear,
    )
    for generator in grid.generators:
        if generator.bus.name not in gfm_buses:
            generator.emt_model.set_parameter_in_model("K", GOVERNOR_K)

    power_flow = PowerFlowDriver(grid=grid, options=ieee9.build_power_flow_options())
    power_flow.run()
    power_flow_3ph = PowerFlowDriver3Ph(grid=grid, options=ieee9.build_power_flow_options())
    power_flow_3ph.run()
    if not power_flow.results.converged or not power_flow_3ph.results.converged:
        raise RuntimeError("IEEE9 power flow did not converge")

    options = ieee9.build_emt_options()
    options.time_step = time_step
    options.simulation_time = simulation_time
    options.tolerance = 1.0e-9
    if integration_method is not None:
        options.integration_method = integration_method
    if initialization_method is not None:
        options.initialization_method = initialization_method

    problem_class = EmtProblemMultilinear if multilinear else EmtProblemDae
    problem = problem_class(
        grid=grid,
        options=options,
        pf_results=power_flow.results,
        pf_results_3ph=power_flow_3ph.results,
    )

    # Device states come from model init_eqs/diff_init_eqs. These predictors
    # only provide the algebraic voltage starting point owned by the network.
    validation.seed_bus_algebraic_predictors(problem, grid, power_flow.results)
    gfm_model = gfm_blocks[0][1]
    return Ieee9Case(grid, problem, power_flow.results, gfm_model)


def signal(case: Ieee9Case, values: np.ndarray, name: str) -> np.ndarray:
    """Return one named signal from the GFM model."""
    result = ieee9._get_signal(case.problem, values, case.gfm_model, name)
    if result is None:
        raise KeyError(f"GFM signal {name!r} was not found")
    return np.asarray(result, dtype=float)
