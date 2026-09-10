"""Time-domain event simulation of the stabilized original IEEE9 EMT model."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt.emt_ieee9 import (
    _find_model_variable_by_name,
    attach_emt_models,
    build_emt_options,
    build_ieee9_grid,
    build_power_flow_options,
)
from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Devices.Events.emt_events_group import EmtEventsGroup
from VeraGridEngine.Simulations.EMT.emt_problem_factory import build_emt_problem
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block
from VeraGridEngine.enumerations import EmtProblemTypes


EVENT_START = 0.10
EVENT_END = 0.20
SPEED_REFERENCE_STEP = 1.0e-3
SIMULATION_TIME = 1.0


def _add_speed_reference_pulse(grid) -> None:
    generator = grid.generators[0]
    parameter = _find_model_variable_by_name(generator.emt_model, "omega_ref")
    if parameter is None:
        raise KeyError("Generator 1 governor has no omega_ref event parameter")
    group = EmtEventsGroup(name="generator_1_speed_reference_pulse")
    grid.add_emt_events_group(group)
    grid.add_emt_event(EmtEvent(device=generator, parameter=parameter, time=EVENT_START,
                                value=1.0 + SPEED_REFERENCE_STEP, group=group))
    grid.add_emt_event(EmtEvent(device=generator, parameter=parameter, time=EVENT_END,
                                value=1.0, group=group))


def _device_series(problem, values: np.ndarray, device: Any, name: str) -> np.ndarray:
    variable = find_name_in_block(name, device.emt_model)
    if variable is None:
        raise KeyError(f"{device.name}: variable '{name}' not found")
    return values[:, int(problem.get_var_idx(variable))]


def main() -> None:
    grid = build_ieee9_grid()
    attach_emt_models(
        grid,
        multilinear_controls=False,
        multilinear_machine=False,
    )
    _add_speed_reference_pulse(grid)

    pf_options = build_power_flow_options()
    balanced_pf = PowerFlowDriver(grid=grid, options=pf_options)
    balanced_pf.run()
    three_phase_pf = PowerFlowDriver3Ph(grid=grid, options=pf_options)
    three_phase_pf.run()
    if not bool(balanced_pf.results.converged) or not bool(three_phase_pf.results.converged):
        raise RuntimeError("IEEE9 power flow failed before the EMT event simulation")

    options = build_emt_options()
    options.problem_type = EmtProblemTypes.CurrentBalance
    options.simulation_time = SIMULATION_TIME
    problem = build_emt_problem(
        grid=grid,
        options=options,
        pf_results=balanced_pf.results,
        pf_results_3ph=three_phase_pf.results,
    )
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=SIMULATION_TIME,
        h=float(options.time_step),
        method=options.integration_method,
    )
    time, values, derivatives, well_initialized, converged = solver.simulate(
        boundary_updater=cast(Any, problem)
    )
    if not bool(well_initialized) or not bool(converged):
        raise RuntimeError(
            f"Event simulation failed: well_initialized={well_initialized}, converged={converged}"
        )

    omega = np.column_stack([
        _device_series(problem, values, generator, "omega_") for generator in grid.generators
    ])
    theta = np.column_stack([
        _device_series(problem, values, generator, "theta_abs_") for generator in grid.generators
    ])
    rotor_angle_deviation = theta - theta[0, :] - 2.0 * np.pi * grid.fBase * time[:, None]
    load_bus = grid.loads[0].bus
    v_a_variable = load_bus.emt_model.out_vars[0]
    v_a = values[:, int(problem.get_var_idx(v_a_variable))]

    directory = Path(__file__).resolve().parent
    csv_path = directory / "ieee9_emt_original_machine_event.csv"
    plot_path = directory / "ieee9_emt_original_machine_event.png"
    stride = max(1, len(time) // 5000)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "omega_g1_pu", "omega_g2_pu", "omega_g3_pu",
                         "delta_theta_g1_rad", "delta_theta_g2_rad", "delta_theta_g3_rad",
                         f"vA_{load_bus.name}_pu"))
        for index in range(0, len(time), stride):
            writer.writerow((time[index], *omega[index, :], *rotor_angle_deviation[index, :], v_a[index]))

    figure, axes = plt.subplots(3, 1, figsize=(10.0, 8.5), sharex=True, constrained_layout=True)
    for generator_index, generator in enumerate(grid.generators):
        axes[0].plot(time, omega[:, generator_index], linewidth=1.1, label=generator.name)
        axes[1].plot(time, rotor_angle_deviation[:, generator_index], linewidth=1.1,
                     label=generator.name)
    axes[2].plot(time, v_a, color="tab:purple", linewidth=0.75, label=f"vA {load_bus.name}")
    for axis in axes:
        axis.axvspan(EVENT_START, EVENT_END, color="tab:orange", alpha=0.16)
        axis.grid(alpha=0.3)
        axis.legend(loc="best")
    axes[0].set_ylabel("omega [pu]")
    axes[1].set_ylabel("angle deviation [rad]")
    axes[2].set_ylabel("phase-A voltage [pu]")
    axes[2].set_xlabel("Time [s]")
    axes[0].set_title("Stabilized IEEE9 EMT response to generator-1 speed-reference pulse")
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    print(f"well_initialized={bool(well_initialized)} converged={bool(converged)} steps={len(time)-1}")
    print(f"event={EVENT_START:.3f}-{EVENT_END:.3f}s omega_ref_G1=1+{SPEED_REFERENCE_STEP:g}pu")
    print(f"max_speed_deviation_pu={np.max(np.abs(omega - 1.0)):.9e}")
    print(f"final_speed_deviation_pu={np.max(np.abs(omega[-1, :] - 1.0)):.9e}")
    print(f"max_relative_angle_deviation_rad={np.max(np.ptp(rotor_angle_deviation, axis=1)):.9e}")
    print(f"csv={csv_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
