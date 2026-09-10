"""Compare the IEEE9 EMT generator with D=0 and D=2.

The two cases are intentionally identical except for the mechanical damping
coefficient.  Excitation is held at its initialized value so the comparison
isolates the swing equation and governor from the provisional AVR/PSS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
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


TIME_STEP = 50.0e-6
SIMULATION_TIME = 4.0


def run_case(
    damping: float,
    freeze_e_qp: bool = False,
) -> tuple[EmtProblemDae, np.ndarray, np.ndarray]:
    grid = build_ieee9_multicircuit()
    attach_baseline_emt_models(
        grid,
        frozen_generator_excitation=True,
        generator_mechanical_damping=damping,
        frozen_generator_e_qp=freeze_e_qp,
    )

    pf = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError("Power flow failed in damping comparison")

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
        raise RuntimeError(f"D={damping:g} simulation failed")
    return problem, time, values


def machine_signals(
    problem: EmtProblemDae, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = next(gen.emt_model for gen in problem.grid.generators if not gen.bus.is_slack)
    omega = next(var for var in model.state_vars if var.name.startswith("omega"))
    te = find_name_in_block("Te", model)
    tm = find_name_in_block("Tm", model)
    if te is None or tm is None:
        raise RuntimeError("Could not find Te and Tm in complete generator model")
    return (
        values[:, int(problem.get_var_idx(omega))],
        values[:, int(problem.get_var_idx(te))],
        values[:, int(problem.get_var_idx(tm))],
    )


def cycle_average(signal: np.ndarray) -> np.ndarray:
    samples = round(0.02 / TIME_STEP)
    return np.convolve(signal, np.ones(samples) / samples, mode="valid")


def save_case_plot(
    damping: float,
    time: np.ndarray,
    omega: np.ndarray,
    te: np.ndarray,
    tm: np.ndarray,
) -> Path:
    """Save one case as soon as it completes."""
    averaged_te = cycle_average(te)
    averaged_tm = cycle_average(tm)
    averaged_time = time[len(time) - len(averaged_te):]
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(time, omega, linewidth=1.0)
    axes[1].plot(averaged_time, averaged_te, label="Te", linewidth=1.0)
    axes[1].plot(averaged_time, averaged_tm, "--", label="Tm", linewidth=1.0)
    axes[2].plot(averaged_time, averaged_te - averaged_tm, linewidth=1.0)
    axes[0].set_ylabel("Speed (p.u.)")
    axes[1].set_ylabel("Torque (p.u.)")
    axes[2].set_ylabel("Te - Tm (p.u.)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title(f"IEEE9 EMT generator — D={damping:g}")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    suffix = str(damping).replace(".", "p")
    output = Path(__file__).with_name(f"ieee9_emt_damping_D{suffix}.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    cases = {}
    for damping in (0.0, 2.0):
        problem, time, values = run_case(damping)
        omega, te, tm = machine_signals(problem, values)
        cases[damping] = (time, omega, te, tm)
        case_plot = save_case_plot(damping, time, omega, te, tm)

        tail = time >= time[-1] - 1.0
        print(
            f"D={damping:g}: omega_end={omega[-1]:.9f}, "
            f"omega_last1s=[{omega[tail].min():.9f}, {omega[tail].max():.9f}], "
            f"mean_Te_minus_Tm={np.mean(te[tail] - tm[tail]):.9e}"
        )
        print(f"plot_D{damping:g}={case_plot}", flush=True)

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.5), sharex=True, constrained_layout=True)
    for damping, (time, omega, te, tm) in cases.items():
        label = f"D={damping:g}"
        axes[0].plot(time, omega, label=label, linewidth=1.0)
        averaged_time = time[len(time) - len(cycle_average(te)):]
        axes[1].plot(averaged_time, cycle_average(te), label=f"Te, {label}", linewidth=1.0)
        axes[1].plot(averaged_time, cycle_average(tm), "--", label=f"Tm, {label}", linewidth=1.0)
        axes[2].plot(
            averaged_time,
            cycle_average(te - tm),
            label=label,
            linewidth=1.0,
        )

    axes[0].set_ylabel("Speed (p.u.)")
    axes[1].set_ylabel("Torque (p.u.)")
    axes[2].set_ylabel("Te - Tm (p.u.)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title("IEEE9 EMT mechanical-damping ablation")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    output = Path(__file__).with_name("ieee9_emt_damping_ablation.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"plot={output}")


if __name__ == "__main__":
    main()
