"""Compare the procedural scheduler and MTI equations on identical ramp samples."""

import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from compare_filter_ripple import INDUCTANCE, RESISTANCE, phase_voltage, sinusoidal_references
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Utils.procedural_logic import ThreePhaseCarrierPwmLogic
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from VeraGridEngine.Utils.Symbolic.symbolic_ml import (
    mti_three_phase_carrier_pwm_internal_carrier_params,
    mti_three_phase_carrier_pwm_scheduled_params,
)

OMEGA = 2 * np.pi * 1000
HALF = np.pi / OMEGA
STOP = 0.004
TOL = 1e-12


def procedural_events(boundaries, samples):
    factory = VarFactory()
    mods = [factory.add_var(f"m_{phase}") for phase in "abc"]
    gates = [factory.add_var(f"g_{phase}") for phase in "abc"]
    omega, phase = (factory.add_var(name) for name in ("omega", "phase"))
    params = np.array([0., 0., 0., OMEGA, 0.])
    block = Block(algebraic_vars=mods, event_dict=dict.fromkeys([*gates, omega, phase], Const(0)))
    problem = SimpleNamespace(
        sys_block=block, get_var_idx=mods.index, get_x0=lambda: samples[0],
        event_params_values=params,
        uid2idx_event_params={var.uid: i for i, var in enumerate([*gates, omega, phase])},
    )
    pwm = ThreePhaseCarrierPwmLogic(
        *(var.name for var in [*mods, *gates, omega, phase])
    )
    pwm.bind(problem)
    events = [(0., params[:3].copy())]
    for index, (boundary, refs) in enumerate(zip(boundaries, samples)):
        # Evaluate just to the right of a known extremum to avoid floor roundoff.
        if index:
            pwm.update(boundary + 1e-14, refs, params)
            events.append((boundary, params[:3].copy()))
        end = boundaries[index + 1] if index + 1 < len(boundaries) else STOP
        cursor = boundary
        while (edge := pwm.get_next_forced_event_time(cursor, end - TOL)) is not None:
            pwm.update(edge, refs, params)
            events.append((edge, params[:3].copy()))
            cursor = edge
    return events


def mti_events(boundaries, samples, scheduled):
    factory = VarFactory()
    if scheduled:
        time_var = factory.add_var("time")
        block, *ports = mti_three_phase_carrier_pwm_scheduled_params(
            factory, time_var, transition_eps=0
        )
        gate_vars, cross_vars = ports[:3], ports[3:6]
        ref_vars = ports[6:9]
        start_var, half_var = ports[9:11]
        before_after, rise = ports[11:14], ports[14]
    else:
        block, *ports = mti_three_phase_carrier_pwm_internal_carrier_params(
            factory, direction_eps=0
        )
        ref_vars = ports[3:6]
        omega, carrier, rise = ports[6:]
        gate_modes = list(block.boolean_guards)[:3]

    events = []
    for index, (boundary, refs) in enumerate(zip(boundaries, samples)):
        rising = index % 2 == 0
        start = boundary if index else -HALF / 2
        end = boundaries[index + 1] if index + 1 < len(boundaries) else STOP
        bindings = {ref.name: value for ref, value in zip(ref_vars, refs)}
        bindings[rise.name] = float(rising)
        if scheduled:
            bindings.update({start_var.name: start, half_var.name: HALF})
        else:
            bindings[omega.name] = OMEGA
            slope = block.state_eqs[0].eval(**bindings)
            initial_carrier = -1.0 if rising else 1.0

        gate = np.empty(3)
        pending = []
        for phase, ref in enumerate(refs):
            if abs(ref) >= 1 - TOL:
                gate[phase] = float(ref > 0)
                continue
            if scheduled:
                edge = block.init_eqs[cross_vars[phase]].eval(**bindings)
                mode = before_after[phase]
                gate_expr = block.init_eqs[gate_vars[phase]]
                bindings[mode.name] = float(boundary >= edge - TOL)
                gate[phase] = gate_expr.eval(**bindings)
                bindings[mode.name] = 1.
                after = gate_expr.eval(**bindings)
            else:
                bindings[gate_modes[phase].name] = 0.
                guard = block.inequalities[phase]

                def residual(t):
                    bindings[carrier.name] = initial_carrier + slope * (t - start)
                    return guard.eval(**bindings)

                edge = brentq(residual, start, start + HALF, xtol=1e-16)
                after = float(not rising)
                gate[phase] = after if boundary >= edge - TOL else float(rising)
            if boundary + TOL < edge < end - TOL:
                pending.append((edge, phase, after))

        events.append((boundary, gate.copy()))
        for edge, phase, after in sorted(pending):
            gate[phase] = after
            events.append((edge, gate.copy()))
    return events


def phase_edges(events, phase):
    return [(time, int(gate[phase])) for (_, prev), (time, gate) in zip(events, events[1:])
            if prev[phase] != gate[phase]]


def current_at(events, grid):
    time = np.array([time for time, _ in events])
    voltage = phase_voltage(np.array([gate for _, gate in events]))
    initial_current = np.zeros_like(voltage)
    for i, dt in enumerate(np.diff(time), start=1):
        gain = -np.expm1(-RESISTANCE * dt / INDUCTANCE)
        initial_current[i] = initial_current[i - 1] * (1 - gain) + gain * voltage[i - 1] / RESISTANCE
    index = np.searchsorted(time, grid, side="right") - 1
    gain = -np.expm1(-RESISTANCE * (grid - time[index]) / INDUCTANCE)
    return initial_current[index] * (1 - gain[:, None]) + gain[:, None] * voltage[index] / RESISTANCE


def compare(name, reference):
    boundaries = np.r_[0., HALF / 2 + HALF * np.arange(8)]
    samples = np.clip(np.column_stack(reference(boundaries, 50., 0.8)), -1., 1.)
    traces = {
        "Procedural": procedural_events(boundaries, samples),
        "Internal-carrier MTI": mti_events(boundaries, samples, scheduled=False),
        "Scheduled-transition MTI": mti_events(boundaries, samples, scheduled=True),
    }
    baseline = traces["Procedural"]
    grid = np.linspace(0, STOP, 40001)
    baseline_current = current_at(baseline, grid)
    figure, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True, layout="constrained")
    for (label, events), style in zip(traces.items(), ["-", "--", ":"]):
        current = current_at(events, grid)
        for phase in range(3):
            expected = phase_edges(baseline, phase)
            actual = phase_edges(events, phase)
            assert [gate for _, gate in expected] == [gate for _, gate in actual]
            assert events[0][1][phase] == baseline[0][1][phase]
            error = max(abs(a[0] - b[0]) for a, b in zip(expected, actual))
            assert error < TOL, (name, label, phase, error)
            print(f"{name} | {label} | {'ABC'[phase]}: {len(actual)} edges, "
                  f"edge error {error:.3g} s, current error "
                  f"{np.max(np.abs(current[:, phase] - baseline_current[:, phase])):.3g} A")
            axes[phase, 0].step(
                [t * 1000 for t, _ in events] + [STOP * 1000],
                [gate[phase] for _, gate in events] + [events[-1][1][phase]],
                where="post", ls=style, label=label,
            )
            axes[phase, 1].plot(grid * 1000, current[:, phase], style)
    for phase, (gate_ax, current_ax) in enumerate(axes):
        gate_ax.set_ylabel(f"Gate {'ABC'[phase]}")
        current_ax.set_ylabel(f"Current {'ABC'[phase]} [A]")
        for ax in (gate_ax, current_ax):
            ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper center", fontsize=8)
    axes[0, 0].set_title("Aligned switching events")
    axes[0, 1].set_title("Common RL filter replay")
    for ax in axes[-1]:
        ax.set_xlabel("Time [ms]")
    output = Path(__file__).resolve().parent / "plots" / f"pwm_aligned_{name}.png"
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
    compare("sinusoidal", sinusoidal_references)
    compare("variable_dq", harness.pwm_references)


if __name__ == "__main__":
    main()
