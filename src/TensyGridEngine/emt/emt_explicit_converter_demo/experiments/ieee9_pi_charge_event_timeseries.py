"""Simulate and plot a 0.2 s charged IEEE-9 GFL setpoint event.

The three originally zero line-charging B values are set to 0.1, as in the
charge-substituted explicit ODE experiment. At 0.05 s the first GFL active
power reference increases by 10%. The system state stays continuous.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_pi_charge_polynomial import extract
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver


@njit(cache=True)
def polynomial_rhs(z, rows, coefficients, modes, offsets):
    """Evaluate sparse multi-affine coefficients without a dense 2^103 tensor."""
    result = np.zeros(len(z))
    for term in range(len(coefficients)):
        product = coefficients[term]
        for ptr in range(offsets[term], offsets[term + 1]):
            product *= z[modes[ptr]]
        result[rows[term]] += product
    return result


def compile_terms(output, *, tolerance=1e-10):
    rows, coefficients, modes, offsets = [], [], [], [0]
    for row, polynomial in enumerate(output):
        for support, value in polynomial.items():
            if abs(value) <= tolerance:
                continue
            if len(support) != len(set(support)):
                raise ValueError("Event model is not multi-affine")
            rows.append(row)
            coefficients.append(value)
            modes.extend(support)
            offsets.append(len(modes))
    return (np.asarray(rows, dtype=np.int64),
            np.asarray(coefficients, dtype=np.float64),
            np.asarray(modes, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64))


def phase_rms(values):
    """Three-phase instantaneous RMS; balanced sinusoids give a flat trace."""
    return np.sqrt(np.mean(values * values, axis=1))


def balanced_groups(problem, dynamic):
    """Find ABC coordinates for reporting zero-sequence drift."""
    names = [problem.get_state_vars()[index].name for index in dynamic]
    groups = []
    for j in range(len(names) - 2):
        if (names[j].endswith("_A") and names[j + 1] == names[j][:-1] + "B"
                and names[j + 2] == names[j][:-1] + "C"):
            groups.append(np.arange(j, j + 3))
    groups += [np.arange(len(dynamic) + 3 * bus, len(dynamic) + 3 * bus + 3)
               for bus in range(9)]
    return groups


def main(*, event_time=0.05, end_time=0.2, power_step=0.10,
         sample_step=1e-4, output_dir=None):
    if not 0 < event_time < end_time:
        raise ValueError("Event time must lie strictly inside the simulation")
    output_dir = Path(output_dir or Path(__file__).resolve().parents[1] / "simulation_data" / "experiments" / "ieee9_pi_charge_event_timeseries")
    output_dir.mkdir(parents=True, exist_ok=True)

    before, before_poly, dynamic, _, _, _ = extract(replacement_B=0.1)
    first_gfl = next(generator for generator in before.grid.generators
                     if not generator.bus.is_slack)
    nominal_power = first_gfl.P / before.grid.Sbase
    after, after_poly, dynamic_after, _, _, _ = extract(
        replacement_B=0.1, first_gfl_power_scale=1.0 + power_step)
    if [state.name for state in before.get_state_vars()] != [state.name for state in after.get_state_vars()]:
        raise RuntimeError("State ordering changed across the event")
    if not np.array_equal(dynamic, dynamic_after):
        raise RuntimeError("Reduced coordinate ordering changed across the event")

    before_terms = compile_terms(before_poly)
    after_terms = compile_terms(after_poly)
    n = len(before.get_state_vars())
    m = len(before.get_algebraic_vars())

    # The DAE reports its raw initial guess at t=0. Its first accepted step
    # provides a charge-consistent starting point for the explicit ODE.
    initial_step = 2.5e-6
    before.options.time_step = initial_step
    solver = build_emt_solver(options=before.options, problem=before,
                              t0=0.0, t_end=initial_step, h=initial_step,
                              method=before.options.integration_method)
    init_time, init_values, _, initialized, converged = solver.simulate(boundary_updater=before)
    if not initialized or not converged or len(init_time) < 2:
        raise RuntimeError("Could not obtain a consistent DAE starting sample")
    z0 = np.r_[init_values[1, dynamic], init_values[1, n + m - 27:n + m]]
    groups = balanced_groups(before, dynamic)
    initial_zero_sequence = max(abs(np.mean(z0[group])) for group in groups)

    # Force the integration to end at the event. The second segment starts
    # from the identical state with the updated vector field.
    pre_times = np.r_[np.arange(init_time[1], event_time, sample_step), event_time]
    post_times = np.r_[event_time, np.arange(event_time + sample_step,
                                             end_time, sample_step), end_time]
    progress = [0.01]

    def run_rhs(terms):
        def rhs(t, z):
            if t >= progress[0]:
                print(f"integrating t={t:.4f} s, max |state|={np.max(np.abs(z)):.3g}", flush=True)
                progress[0] += 0.01
            return polynomial_rhs(z, *terms)
        return rhs

    pre = solve_ivp(run_rhs(before_terms),
                    (init_time[1], event_time), z0, t_eval=pre_times,
                    method="BDF", rtol=1e-7, atol=1e-9, max_step=5e-5)
    if not pre.success:
        raise RuntimeError(f"Pre-event integration failed: {pre.message}")
    print(f"event reached at {event_time:g} s", flush=True)
    post = solve_ivp(run_rhs(after_terms),
                     (event_time, end_time), pre.y[:, -1], t_eval=post_times,
                     method="BDF", rtol=1e-7, atol=1e-9, max_step=5e-5)
    if not post.success:
        raise RuntimeError(f"Post-event integration failed: {post.message}")
    time = np.r_[pre.t, post.t[1:]]
    state = np.column_stack((pre.y, post.y[:, 1:])).T

    # Voltage coordinates follow the non-charge states in bus order, ABC.
    voltage = state[:, len(dynamic):].reshape(-1, 9, 3)
    current_indices = []
    state_names = [variable.name for variable in before.get_state_vars()]
    for generator in ("gen 1", "gen 2"):
        indices = [int(np.flatnonzero(np.array(state_names) == f"i_network_{generator}_{phase}")[0])
                   for phase in "ABC"]
        current_indices.append([int(np.flatnonzero(dynamic == index)[0]) for index in indices])
    current_rms = np.column_stack([phase_rms(state[:, indices]) for indices in current_indices])
    voltage_rms = np.column_stack([phase_rms(voltage[:, bus]) for bus in (1, 2, 5)])
    event_index = np.searchsorted(time, event_time)
    max_zero_sequence = max(float(np.max(abs(np.mean(state[:, group], axis=1))))
                            for group in groups)
    active_power = np.column_stack([
        np.sum(voltage[:, bus] * state[:, indices], axis=1) / 3.0
        for bus, indices in zip((1, 2), current_indices)
    ])

    data_path = output_dir / "ieee9_pi_charge_event_timeseries.npz"
    plot_path = output_dir / "ieee9_pi_charge_event_timeseries.png"
    np.savez_compressed(data_path, time=time, state=state,
                        bus_voltage_rms=voltage_rms, converter_current_rms=current_rms,
                        converter_active_power=active_power,
                        event_time=event_time, power_step=power_step)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    for column, bus in enumerate((1, 2, 5)):
        axes[0].plot(time, voltage_rms[:, column], label=f"bus {bus}")
    axes[0].set_ylabel("3-phase voltage RMS (pu)")
    axes[0].legend(loc="best")
    for column, generator in enumerate(("gen 1", "gen 2")):
        delta = 1000.0 * (current_rms[:, column] - current_rms[event_index, column])
        axes[1].plot(time, delta, label=generator)
    axes[1].set_ylabel("Change in GFL current RMS (milli-pu)")
    axes[1].legend(loc="best")
    axes[2].plot(time, active_power[:, 0], label="gen 1 measured")
    axes[2].plot(time, active_power[:, 1], label="gen 2 measured")
    axes[2].step(time, np.where(time < event_time, nominal_power,
                                nominal_power * (1.0 + power_step)),
                 where="post", linestyle="--", label="gen 1 reference")
    axes[2].set_ylabel("Generated active power (pu)")
    axes[2].legend(loc="best")
    axes[3].plot(time, voltage[:, 1, 0], label="bus 1, phase A")
    axes[3].plot(time, voltage[:, 2, 0], label="bus 2, phase A", alpha=0.8)
    axes[3].set_ylabel("Phase-A voltage (pu)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="best")
    for axis in axes:
        axis.axvline(event_time, color="black", linestyle="--", linewidth=1)
        axis.grid(alpha=0.25)
    fig.suptitle(f"Modified IEEE-9 EMT: gen 1 P reference +{100 * power_step:g}% at {event_time:g} s")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"samples={len(time)}, interval={time[0]:.7g} to {time[-1]:.3f} s")
    print(f"event: first GFL P reference +{100 * power_step:g}% at {event_time:g} s")
    print(f"integrator calls: before={pre.nfev}, after={post.nfev}")
    print(f"ABC groups={len(groups)}, initial zero-sequence={initial_zero_sequence:.3e}, "
          f"max zero-sequence over run={max_zero_sequence:.3e}")
    print(f"bus 1 voltage RMS: pre-event={voltage_rms[np.searchsorted(time, event_time)-1, 0]:.6g}, "
          f"final={voltage_rms[-1, 0]:.6g}")
    print(f"gen 1 current RMS: pre-event={current_rms[np.searchsorted(time, event_time)-1, 0]:.6g}, "
          f"final={current_rms[-1, 0]:.6g}")
    print(f"gen 1 active power: pre-event={active_power[np.searchsorted(time, event_time)-1, 0]:.6g}, "
          f"final={active_power[-1, 0]:.6g} pu")
    print(f"plot={plot_path}")
    print(f"data={data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-time", type=float, default=0.05)
    parser.add_argument("--end-time", type=float, default=0.2)
    parser.add_argument("--power-step", type=float, default=0.10)
    parser.add_argument("--sample-step", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path)
    main(**vars(parser.parse_args()))
