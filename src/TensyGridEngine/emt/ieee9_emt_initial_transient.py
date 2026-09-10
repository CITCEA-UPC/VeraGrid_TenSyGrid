"""Inspect the first 100 ms of the IEEE9 EMT generator initialization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ieee9_emt_damping_ablation as ablation
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block


TIME_STEP = 5.0e-6
SIMULATION_TIME = 0.04
DAMPING = 0.0


def main() -> None:
    # Reuse the exact damping-ablation case construction, changing only its
    # numerical horizon.  The governor remains active and excitation frozen.
    ablation.TIME_STEP = TIME_STEP
    ablation.SIMULATION_TIME = SIMULATION_TIME
    problem, time, values = ablation.run_case(DAMPING)
    omega, te, tm = ablation.machine_signals(problem, values)
    model = next(gen.emt_model for gen in problem.grid.generators if not gen.bus.is_slack)
    pe_var = find_name_in_block("p_e", model)
    current_vars = [find_name_in_block(f"i_{phase}", model) for phase in ("A", "B", "C")]
    if pe_var is None or any(var is None for var in current_vars):
        raise RuntimeError("Could not find generator terminal power or phase currents")
    pe = values[:, int(problem.get_var_idx(pe_var))]
    currents = [values[:, int(problem.get_var_idx(var))] for var in current_vars]
    air_gap_power = te * omega
    torque_error = te - tm
    omega_slope = np.gradient(omega, time)

    cycle_samples = round(0.02 / TIME_STEP)
    cycle_te = np.convolve(te, np.ones(cycle_samples) / cycle_samples, mode="valid")
    cycle_tm = np.convolve(tm, np.ones(cycle_samples) / cycle_samples, mode="valid")
    cycle_pe = np.convolve(pe, np.ones(cycle_samples) / cycle_samples, mode="valid")
    cycle_air_gap_power = np.convolve(
        air_gap_power, np.ones(cycle_samples) / cycle_samples, mode="valid"
    )
    cycle_time = time[len(time) - len(cycle_te):]

    sample_times = list((0.0, TIME_STEP, 1.0e-3, 20.0e-3, 40.0e-3))
    if SIMULATION_TIME not in sample_times:
        sample_times.append(SIMULATION_TIME)
    for sample_time in sample_times:
        idx = int(np.argmin(np.abs(time - sample_time)))
        print(
            f"t={time[idx]:.9e}, omega={omega[idx]:.12f}, "
            f"Te={te[idx]:.12f}, Tm={tm[idx]:.12f}, "
            f"Pe={pe[idx]:.12f}, Te_omega={air_gap_power[idx]:.12f}, "
            f"Te_minus_Tm={torque_error[idx]:.12f}, "
            f"domega_dt={omega_slope[idx]:.12f}"
        )

    print(
        f"first_step: delta_omega={omega[1] - omega[0]:.12e}, "
        f"delta_Te={te[1] - te[0]:.12e}, delta_Tm={tm[1] - tm[0]:.12e}"
    )
    print(
        f"first_cycle_mean: Pe={cycle_pe[0]:.12f}, "
        f"Te_omega={cycle_air_gap_power[0]:.12f}, "
        f"difference={cycle_air_gap_power[0] - cycle_pe[0]:.12f}"
    )
    print(
        f"last_cycle_mean: Pe={cycle_pe[-1]:.12f}, "
        f"Te_omega={cycle_air_gap_power[-1]:.12f}, "
        f"difference={cycle_air_gap_power[-1] - cycle_pe[-1]:.12f}"
    )

    time_ms = time * 1.0e3
    fig, axes = plt.subplots(5, 1, figsize=(10.0, 12.0), sharex=True, constrained_layout=True)
    axes[0].plot(time_ms, omega)
    axes[0].set_ylabel("Speed (p.u.)")
    axes[1].plot(time_ms, te, alpha=0.35, linewidth=0.7, label="Te instantaneous")
    axes[1].plot(time_ms, tm, "--", linewidth=1.0, label="Tm instantaneous")
    axes[1].plot(cycle_time * 1.0e3, cycle_te, linewidth=1.2, label="Te 20 ms average")
    axes[1].plot(cycle_time * 1.0e3, cycle_tm, "--", linewidth=1.2, label="Tm 20 ms average")
    axes[1].set_ylabel("Torque (p.u.)")
    axes[1].legend(ncol=2)
    axes[2].plot(cycle_time * 1.0e3, cycle_pe, label="terminal Pe")
    axes[2].plot(cycle_time * 1.0e3, cycle_air_gap_power, label="Te·ω")
    axes[2].set_ylabel("Power (p.u.)")
    axes[2].legend()
    axes[3].plot(time_ms, torque_error)
    axes[3].set_ylabel("Te - Tm (p.u.)")
    axes[4].plot(time_ms, omega_slope)
    axes[4].set_ylabel("dω/dt (p.u./s)")
    axes[4].set_xlabel("Time (ms)")
    duration_ms = int(round(1.0e3 * SIMULATION_TIME))
    axes[0].set_title(
        f"IEEE9 EMT initial transient ({duration_ms} ms) — D=0, governor active, excitation frozen"
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)

    output = Path(__file__).with_name(
        f"ieee9_emt_initial_{duration_ms}ms_power_balance_D0.png"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"plot={output}")

    current_fig, current_axes = plt.subplots(
        2, 1, figsize=(10.0, 7.0), sharex=True, constrained_layout=True
    )
    for phase, current in zip(("A", "B", "C"), currents):
        current_axes[0].plot(time_ms, current, label=f"i{phase}", linewidth=0.9)
    current_magnitude = np.sqrt(sum(current ** 2 for current in currents))
    current_axes[1].plot(time_ms, current_magnitude, color="black", linewidth=1.0)
    current_axes[0].set_ylabel("Phase current (p.u.)")
    current_axes[1].set_ylabel("sqrt(iA²+iB²+iC²) (p.u.)")
    current_axes[1].set_xlabel("Time (ms)")
    current_axes[0].set_title("IEEE9 EMT generator terminal currents — D=0")
    current_axes[0].legend(ncol=3)
    for axis in current_axes:
        axis.grid(True, alpha=0.3)
    current_output = Path(__file__).with_name(
        f"ieee9_emt_initial_{duration_ms}ms_generator_currents_D0.png"
    )
    current_fig.savefig(current_output, dpi=180)
    plt.close(current_fig)
    print(f"current_plot={current_output}")

    machine_signal_names = (
        "psi_d_", "psi_q_", "e_qp_", "e_dp_", "psi_pp_d_", "psi_pp_q_",
        "v_d_", "v_q_", "i_d_", "i_q_", "IRPu",
    )
    machine_signals = {}
    print("MACHINE_STATE_AND_DQ_CHANGES")
    for name in machine_signal_names:
        variable = find_name_in_block(name, model)
        if variable is None:
            continue
        signal = values[:, int(problem.get_var_idx(variable))]
        machine_signals[name] = signal
        print(
            f"{name}: initial={signal[0]:+.12e}, end={signal[-1]:+.12e}, "
            f"delta={signal[-1] - signal[0]:+.12e}, "
            f"range={signal.max() - signal.min():.12e}"
        )

    state_fig, state_axes = plt.subplots(3, 1, figsize=(10.0, 8.5), sharex=True, constrained_layout=True)
    for name in ("psi_d_", "psi_q_", "psi_pp_d_", "psi_pp_q_"):
        signal = machine_signals.get(name)
        if signal is not None:
            state_axes[0].plot(time_ms, signal - signal[0], label=name)
    for name in ("e_qp_", "e_dp_", "IRPu"):
        signal = machine_signals.get(name)
        if signal is not None:
            state_axes[1].plot(time_ms, signal - signal[0], label=name)
    for name in ("v_d_", "v_q_", "i_d_", "i_q_"):
        signal = machine_signals.get(name)
        if signal is not None:
            state_axes[2].plot(time_ms, signal - signal[0], label=name)
    state_axes[0].set_ylabel("Flux-state delta")
    state_axes[1].set_ylabel("EMF/field delta")
    state_axes[2].set_ylabel("dq delta")
    state_axes[2].set_xlabel("Time (ms)")
    state_axes[0].set_title("IEEE9 EMT synchronous-machine initialization drift")
    for axis in state_axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=3)
    state_output = Path(__file__).with_name(
        f"ieee9_emt_initial_{duration_ms}ms_machine_state_drift_D0.png"
    )
    state_fig.savefig(state_output, dpi=180)
    plt.close(state_fig)
    print(f"machine_state_plot={state_output}")


if __name__ == "__main__":
    main()
