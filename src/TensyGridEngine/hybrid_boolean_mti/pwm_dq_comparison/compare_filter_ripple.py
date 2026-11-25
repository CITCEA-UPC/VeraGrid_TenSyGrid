"""Replay each PWM comparison through the same three-phase RL filter."""

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

from compare_soft_pwm import soft_gates

STEP = 1e-7
PERIOD_SAMPLES = 10000
LEG_VOLTAGE = 400.0
RESISTANCE = 0.2
INDUCTANCE = 2e-3


def replay_current(voltage, step):
    decay = np.exp(-RESISTANCE * step / INDUCTANCE)
    gain = -np.expm1(-RESISTANCE * step / INDUCTANCE) / RESISTANCE
    return lfilter([0, gain], [1, -decay], voltage, axis=0)


def hold(time, values, grid):
    indices = np.clip(np.searchsorted(time, grid, side="right") - 1, 0, len(time) - 1)
    return values[indices]


def phase_voltage(gates):
    leg_voltage = LEG_VOLTAGE * (2 * gates - 1)
    return leg_voltage - leg_voltage.mean(axis=1, keepdims=True)


def compare(harness, case):
    grid = np.arange(4 * PERIOD_SAMPLES + 1) * STEP
    refs = np.column_stack(harness.pwm_references(grid, 50.0, 0.8))
    gates = {"Compatible soft PWM": soft_gates(grid, refs)}
    for procedural, label in [(True, "Procedural hard PWM"), (False, "Live comparator")]:
        _, _, time, gate_trace, initialized, converged, _ = harness.run_solver_backed_case(procedural)
        if not (initialized and converged):
            raise RuntimeError(f"{case}: {label} solver did not converge")
        gates[label] = hold(time, gate_trace, grid)
    for run, label in [
        (harness.run_back_euler_mti_block_case, "Internal-carrier MTI"),
        (harness.run_back_euler_mti_scheduled_block_case, "Scheduled-transition MTI"),
    ]:
        time, values, modes, initialized, converged, _ = run()
        if not (initialized and converged):
            raise RuntimeError(f"{case}: {label} solver did not converge")
        waveform = modes[:, :3] if run is harness.run_back_euler_mti_block_case else values
        gates[label] = hold(time, waveform, grid)

    order = ["Procedural hard PWM", "Compatible soft PWM", "Live comparator",
             "Internal-carrier MTI", "Scheduled-transition MTI"]
    plot_currents(gates, grid, case, order)


def plot_currents(gates, grid, case, order):
    step = grid[1] - grid[0]
    styles = ["-", "--", "-", "-.", ":"]
    figure, axes = plt.subplots(3, 2, figsize=(12, 8), layout="constrained")
    for label, style in zip(order, styles):
        voltage = phase_voltage(gates[label])
        current = replay_current(voltage, step)
        cycles = voltage[:-1].reshape(4, PERIOD_SAMPLES, 3)
        average = np.repeat(cycles.mean(axis=1), PERIOD_SAMPLES, axis=0)
        average = np.vstack([average, average[-1]])
        ripple = current - replay_current(average, step)
        # Exclude the first carrier cycle from the peak-to-peak summary.
        ripple_range = np.ptp(ripple[PERIOD_SAMPLES:], axis=0)
        print(f"{case} | {label} | ripple peak-to-peak A/B/C [A]: "
              + " / ".join(f"{value:.2f}" for value in ripple_range))
        for phase, (full, detail) in enumerate(axes):
            full.plot(grid * 1000, current[:, phase], style, lw=1.1, label=label)
            detail.plot(grid * 1000, ripple[:, phase], style, lw=1.1)
            full.set_ylabel(f"Phase {'ABC'[phase]} [A]")
            detail.set_xlim(grid[PERIOD_SAMPLES] * 1000, grid[3 * PERIOD_SAMPLES] * 1000)
            full.grid(alpha=0.22)
            detail.grid(alpha=0.22)
    axes[0, 0].set_title("Filter current")
    axes[0, 1].set_title("Current ripple relative to cycle-average voltage")
    for ax in axes[-1]:
        ax.set_xlabel("Time [ms]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=3)
    output = Path(__file__).resolve().parent / "plots" / f"pwm_filter_ripple_{case}.png"
    figure.savefig(output, dpi=170)
    plt.close(figure)


def sinusoidal_references(time, frequency, modulation_index):
    angle = 2 * np.pi * frequency * np.asarray(time)
    return tuple(modulation_index * np.sin(angle + shift) for shift in (0, -2*np.pi/3, 2*np.pi/3))


def compare_slow_tie_break(harness):
    config = dict(f_ref=1.0, f_sw=20.0, t_end=0.1, h=2.5e-3)
    grid = np.arange(4 * PERIOD_SAMPLES + 1) * (0.025 / PERIOD_SAMPLES)
    _, _, time, values, initialized, converged, _ = harness.run_solver_backed_case(True, **config)
    if not (initialized and converged):
        raise RuntimeError("Slow procedural PWM solver did not converge")
    gates = {"Procedural hard PWM": hold(time, values, grid)}
    for bias, label in [(0, "Internal-carrier MTI, zero bias"),
                        (1e-8, "Internal-carrier MTI, directional bias")]:
        time, _, modes, initialized, converged, _ = harness.run_back_euler_mti_block_case(
            direction_eps=bias, **config
        )
        if not (initialized and converged):
            raise RuntimeError(f"Slow {label} solver did not converge")
        gates[label] = hold(time, modes[:, :3], grid)

    # The original slow test spans two full carrier periods.
    figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for (label, gate), style in zip(gates.items(), ["-", "--", ":"]):
        voltage = phase_voltage(gate)
        average = voltage[:-1].reshape(2, 2 * PERIOD_SAMPLES, 3).mean(axis=1)
        average = np.repeat(average, 2 * PERIOD_SAMPLES, axis=0)
        average = np.vstack([average, average[-1]])
        step = grid[1] - grid[0]
        ripple = replay_current(voltage, step) - replay_current(average, step)
        for phase, ax in enumerate(axes):
            ax.plot(grid * 1000, ripple[:, phase], style, label=label)
            ax.set_ylabel(f"Phase {'ABC'[phase]} [A]")
            ax.grid(alpha=0.22)
    axes[0].legend()
    axes[0].set_title("20 Hz tie-break test, current ripple")
    axes[-1].set_xlabel("Time [ms]")
    output = Path(__file__).resolve().parent / "plots" / "pwm_filter_ripple_slow_tie_break.png"
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("variable_dq_harness", args.harness)
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness
    spec.loader.exec_module(harness)
    compare(harness, "variable_dq")
    harness.pwm_references = sinusoidal_references
    compare(harness, "sinusoidal")
    compare_slow_tie_break(harness)


if __name__ == "__main__":
    main()
