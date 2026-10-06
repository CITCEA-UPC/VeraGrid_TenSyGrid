"""Run IEEE9 with Thevenin sources at both retained generator buses."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block

from ieee9_emt_from_scratch import (
    attach_baseline_emt_models,
    build_emt_options,
    build_ieee9_multicircuit,
    build_power_flow_options,
)


TIME_STEP = 5.0e-6
SIMULATION_TIME = 5.0e-3


def main() -> None:
    grid = build_ieee9_multicircuit()
    attach_baseline_emt_models(
        grid,
        use_thevenin_for_dynamic_generators=True,
        use_constant_power_loads=False,
    )
    pf = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf3.run()
    options = build_emt_options()
    options.time_step = TIME_STEP
    options.simulation_time = SIMULATION_TIME
    problem = EmtProblemDae(
        grid=grid,
        options=options,
        pf_results_3ph=pf3.results,
        pf_results=pf.results,
    )
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=SIMULATION_TIME,
        h=TIME_STEP,
        method=options.integration_method,
    )
    time, values, _differentials, initialized, converged = solver.simulate(
        boundary_updater=cast(Any, problem)
    )
    if not initialized or not converged or not np.isfinite(values).all():
        raise RuntimeError("Thevenin-source ablation failed")

    for generator in grid.generators:
        print(generator.name)
        current_arrays = []
        for phase in "ABC":
            variable = find_name_in_block(f"i_{phase}", generator.emt_model)
            idx = int(problem.get_var_idx(variable))
            waveform = values[:, idx]
            current_arrays.append(waveform)
            print(
                f"  i{phase}: x0={waveform[0]:+.9f}, x1ms={waveform[round(1e-3/TIME_STEP)]:+.9f}, "
                f"x5ms={waveform[-1]:+.9f}, range={np.ptp(waveform):.9f}"
            )
        magnitude = np.sqrt((2.0 / 3.0) * sum(current ** 2 for current in current_arrays))
        print(
            f"  positive-sequence peak magnitude: initial={magnitude[0]:.9f}, "
            f"1ms={magnitude[round(1e-3/TIME_STEP)]:.9f}, final={magnitude[-1]:.9f}, "
            f"range={np.ptp(magnitude):.9f}"
        )
    print(f"samples={len(time)}, final_time={time[-1]:.9f}")


if __name__ == "__main__":
    main()
