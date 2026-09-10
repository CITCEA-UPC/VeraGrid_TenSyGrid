"""IEEE9 EMT case using the multilinear synchronous-generator formulation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Devices.Events.emt_events_group import EmtEventsGroup
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Templates.Emt.generator_sauer_pai_emt_multilinear_template import (
    get_complete_generator_template_emt_multilinear,
)
from VeraGridEngine.Utils.Symbolic.block import Block, Var, find_name_in_block

from ieee9_emt_from_scratch import (
    USE_CONVENTIONAL_THREE_PHASE_BASE,
    attach_baseline_emt_models,
    build_emt_options,
    build_ieee9_multicircuit,
    build_power_flow_options,
)


TIME_STEP = 5.0e-6
SIMULATION_TIME = 0.04
LOAD_STEP_TIME = 0.1
LOAD_STEP_SCALE = 1.05


def get_ieee9_multilinear_generator(vf: Any, name: str) -> Block:
    """Build the same controlled Sauer-Pai machine with multilinear Park transforms."""
    return get_complete_generator_template_emt_multilinear(
        vf=vf,
        name=name,
        conventional_three_phase_base=USE_CONVENTIONAL_THREE_PHASE_BASE,
        frozen_excitation=True,
    ).block


def add_load_step_event(grid: Any, load_index: int, event_time: float, scale: float) -> str:
    """Scale P and Q of one three-phase ZIP load at ``event_time``."""
    if not 0 <= load_index < len(grid.loads):
        raise IndexError(f"Load index {load_index} is outside [0, {len(grid.loads) - 1}]")

    load = grid.loads[load_index]
    model = load.emt_model
    event_group = EmtEventsGroup(name=f"load_{load_index + 1}_pq_step")
    grid.add_emt_events_group(event_group)

    for quantity, total_value in (("P", load.P), ("Q", load.Q)):
        for phase_index, phase in enumerate("ABC", start=1):
            phase_value_raw = getattr(load, f"{quantity}{phase_index}", None)
            phase_value = total_value / 3.0 if phase_value_raw is None else phase_value_raw
            parameter_name = f"{quantity}0_{phase}"
            parameter = find_name_in_block(parameter_name, model)
            if parameter is None:
                parameter = next(
                    (variable for variable in model.api_obj_mapping.values()
                     if variable is not None and variable.name == parameter_name),
                    None,
                )
            if parameter is None:
                raise KeyError(f"Could not find {parameter_name} for load event")
            # PF-mapped load powers are constants unless explicitly promoted to
            # event parameters before the EMT problem clones the device model.
            model.event_dict[parameter] = grid.var_factory.add_const(float(phase_value / grid.Sbase))
            grid.add_emt_event(
                EmtEvent(
                    name=f"{load.name} {quantity}{phase} step",
                    device=load,
                    parameter=parameter,
                    time=event_time,
                    value=float(scale * phase_value / grid.Sbase),
                    group=event_group,
                    force_step_alignment=True,
                )
            )
    return load.name


def run_simulation(
    simulation_time: float = SIMULATION_TIME,
    time_step: float = TIME_STEP,
    load_index: int = 0,
    load_step_time: float = LOAD_STEP_TIME,
    load_step_scale: float = LOAD_STEP_SCALE,
    generator_builder: Any = get_ieee9_multilinear_generator,
) -> tuple[EmtProblemDae, np.ndarray, np.ndarray, str]:
    grid = build_ieee9_multicircuit()
    attach_baseline_emt_models(
        grid,
        dynamic_generator_builder=generator_builder,
    )
    event_load_name = add_load_step_event(grid, load_index, load_step_time, load_step_scale)
    pf = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError("IEEE9 power flow failed")

    options = build_emt_options()
    options.time_step = time_step
    options.simulation_time = simulation_time
    problem = EmtProblemDae(
        grid=grid,
        options=options,
        pf_results_3ph=pf3.results,
        pf_results=pf.results,
    )
    problem.set_events_group(grid.emt_events_groups[-1])
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=simulation_time,
        h=time_step,
        method=options.integration_method,
    )
    time, values, _differentials, initialized, converged = solver.simulate(
        boundary_updater=cast(Any, problem)
    )
    if not initialized or not converged or not np.isfinite(values).all():
        raise RuntimeError(
            "IEEE9 multilinear EMT simulation failed: "
            f"initialized={initialized}, converged={converged}, "
            f"finite={np.isfinite(values).all()}, samples={len(time)}"
        )
    return problem, time, values, event_load_name


def get_signal(problem: EmtProblemDae, values: np.ndarray, model: Block, name: str) -> np.ndarray:
    variable: Var | None = find_name_in_block(name, model)
    if variable is None:
        variable = next((candidate for candidate in model.get_all_vars() if candidate.name.startswith(name)), None)
    if variable is None:
        raise RuntimeError(f"Could not find multilinear signal {name}")
    return values[:, int(problem.get_var_idx(variable))]


def trailing_cycle_mean(signal: np.ndarray, samples_per_cycle: int) -> np.ndarray:
    """Return a trailing one-cycle mean, leaving the incomplete prefix as NaN."""
    result = np.full(signal.shape, np.nan, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(signal, dtype=float)))
    result[samples_per_cycle - 1:] = (
        cumulative[samples_per_cycle:] - cumulative[:-samples_per_cycle]
    ) / samples_per_cycle
    return result


def run_case(generator_builder: Any, case_title: str, output_stem: str) -> None:
    """Run and plot one classic or multilinear generator load-event case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-time", type=float, default=SIMULATION_TIME, help="Simulation horizon in seconds")
    parser.add_argument("--time-step", type=float, default=TIME_STEP, help="EMT time step in seconds")
    parser.add_argument("--load-index", type=int, default=0, help="Zero-based index of the disturbed load")
    parser.add_argument("--load-step-time", type=float, default=LOAD_STEP_TIME, help="Load-step time in seconds")
    parser.add_argument("--load-step-scale", type=float, default=LOAD_STEP_SCALE, help="Post-event P/Q multiplier")
    args = parser.parse_args()
    problem, time, values, event_load_name = run_simulation(
        args.simulation_time,
        args.time_step,
        args.load_index,
        args.load_step_time,
        args.load_step_scale,
        generator_builder,
    )
    generator = next(gen for gen in problem.grid.generators if not gen.bus.is_slack)
    model = generator.emt_model
    omega = get_signal(problem, values, model, "omega")
    te = get_signal(problem, values, model, "Te")
    tm = get_signal(problem, values, model, "Tm")
    currents = [get_signal(problem, values, model, f"i_{phase}") for phase in "ABC"]
    try:
        u_cos = get_signal(problem, values, model, "u_cos")
        u_sin = get_signal(problem, values, model, "u_sin")
    except RuntimeError:
        theta = get_signal(problem, values, model, "theta")
        u_cos = np.cos(theta)
        u_sin = np.sin(theta)
    trig_norm = u_cos * u_cos + u_sin * u_sin
    event_load = problem.grid.loads[args.load_index]
    load_currents = [get_signal(problem, values, event_load.emt_model, f"i_{phase}") for phase in "ABC"]
    load_voltages = [
        values[:, int(problem.get_var_idx(event_load.bus.emt_model.out_vars[index]))]
        for index in range(3)
    ]
    v_a, v_b, v_c = load_voltages
    i_a, i_b, i_c = load_currents
    v_alpha = (2.0 / 3.0) * (v_a - 0.5 * v_b - 0.5 * v_c)
    v_beta = (np.sqrt(3.0) / 3.0) * (v_b - v_c)
    i_alpha = (2.0 / 3.0) * (i_a - 0.5 * i_b - 0.5 * i_c)
    i_beta = (np.sqrt(3.0) / 3.0) * (i_b - i_c)
    load_p_instantaneous = -(v_a * i_a + v_b * i_b + v_c * i_c) / 3.0
    load_q_instantaneous = 0.5 * (v_alpha * i_beta - v_beta * i_alpha)
    samples_per_cycle = max(1, int(round(1.0 / (problem.grid.fBase * args.time_step))))
    load_p_cycle = trailing_cycle_mean(load_p_instantaneous, samples_per_cycle)
    load_q_cycle = trailing_cycle_mean(load_q_instantaneous, samples_per_cycle)
    voltage_rms = np.sqrt(
        trailing_cycle_mean((v_a * v_a + v_b * v_b + v_c * v_c) / 3.0, samples_per_cycle)
    )
    torque_gap = te - tm
    torque_gap_cycle = trailing_cycle_mean(torque_gap, samples_per_cycle)
    event_index = int(np.searchsorted(time, args.load_step_time, side="left"))
    before_index = max(0, event_index - 1)
    after_index = min(len(time) - 1, event_index + 1)
    print(
        f"load_power_before={load_p_instantaneous[before_index]:+.9e}, "
        f"load_power_after={load_p_instantaneous[after_index]:+.9e}"
    )
    print(
        f"final_cycle_P={load_p_cycle[-1]:+.9e}, final_cycle_Q={load_q_cycle[-1]:+.9e}, "
        f"final_cycle_Vrms={voltage_rms[-1]:+.9e}, "
        f"final_cycle_Te_minus_Tm={torque_gap_cycle[-1]:+.9e}"
    )

    print(
        f"initialized=True, converged=True, steps={len(time) - 1}, "
        f"omega_end={omega[-1]:.12f}, Te_minus_Tm_end={te[-1] - tm[-1]:+.12e}, "
        f"max_trig_norm_error={np.max(np.abs(trig_norm - 1.0)):.12e}, "
        f"event_load={event_load_name!r}, event_time={args.load_step_time:.6g}, "
        f"load_scale={args.load_step_scale:.6g}"
    )
    for idx in sorted(set((0, min(1, len(time) - 1), len(time) - 1))):
        print(
            f"t={time[idx]:.9e}, omega={omega[idx]:.12f}, "
            f"Te={te[idx]:+.12e}, Tm={tm[idx]:+.12e}, "
            f"iA={currents[0][idx]:+.12e}"
        )

    time_ms = 1.0e3 * time
    fig, axes = plt.subplots(5, 1, figsize=(10.0, 12.0), sharex=True, constrained_layout=True)
    axes[0].plot(time_ms, 1.0e6 * (omega - 1.0))
    axes[0].set_ylabel("omega−1 (ppm)")
    axes[1].plot(time_ms, torque_gap, alpha=0.35, label="instantaneous")
    axes[1].plot(time_ms, torque_gap_cycle, label="1-cycle mean")
    axes[1].set_ylabel("Te−Tm (p.u.)")
    axes[1].legend()
    axes[2].plot(time_ms, load_p_cycle, label="P")
    axes[2].plot(time_ms, load_q_cycle, label="Q")
    axes[2].set_ylabel("Load power (p.u.)")
    axes[2].legend(ncol=2)
    axes[3].plot(time_ms, voltage_rms)
    axes[3].set_ylabel("Load-bus Vrms (p.u.)")
    axes[4].plot(time_ms, trig_norm - 1.0)
    axes[4].set_ylabel("u_cos²+u_sin²−1")
    axes[4].set_xlabel("Time (ms)")
    axes[0].set_title(f"IEEE9 EMT — {case_title}")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        if 0.0 <= args.load_step_time <= args.simulation_time:
            axis.axvline(1.0e3 * args.load_step_time, color="black", linestyle=":", alpha=0.65)
    horizon_ms = args.simulation_time * 1.0e3
    horizon_label = f"{horizon_ms:g}".replace(".", "p")
    event_suffix = f"_load_step_{args.load_step_scale:g}x" if args.load_step_time <= args.simulation_time else ""
    time_step_us = args.time_step * 1.0e6
    time_step_label = f"{time_step_us:g}".replace(".", "p")
    output = Path(__file__).with_name(
        f"{output_stem}_{horizon_label}ms{event_suffix}_dt{time_step_label}us.png"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"plot={output}")


def main() -> None:
    run_case(
        generator_builder=get_ieee9_multilinear_generator,
        case_title="multilinear synchronous generator",
        output_stem="ieee9_emt_multilinear",
    )


if __name__ == "__main__":
    main()
