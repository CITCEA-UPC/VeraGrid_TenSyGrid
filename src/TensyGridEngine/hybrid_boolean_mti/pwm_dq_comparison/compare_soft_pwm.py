"""Compare the compatible soft PWM comparator on the variable-dq test signals."""

import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.problems.emt_problem_template import EmtProblemTemplate
from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_soft_pwm_compatible_template import (
    get_bridge_2level_3ph_emt_soft_pwm_compatible_template,
)


def soft_gates(time, references):
    """Evaluate the template's carrier, sample-and-hold, and gate equations."""
    name = "soft_comparison"
    block = get_bridge_2level_3ph_emt_soft_pwm_compatible_template(VarFactory(), name=name).block
    sampler, = block.procedural_logic
    modulation = block.out_vars[6:9]
    sample_modes = list(block.mode_dict)
    omega_sw, carrier_phase, enable = block.in_vars[-3:]
    carrier, = [var for var in block.event_dict if var.name == f"carrier_pwm_{name}"]
    parameters = [*sample_modes, omega_sw, carrier_phase]
    sampler.bind(SimpleNamespace(
        sys_block=block,
        uid2idx_vars={var.uid: i for i, var in enumerate(modulation)},
        uid2idx_event_params={var.uid: i for i, var in enumerate(parameters)},
    ))

    values = np.array([0.0, 0.0, 0.0, 2 * np.pi * 1000, 0.0])
    bindings = {enable.name: 1.0}
    carrier_expr = block.event_dict[carrier]
    gate_exprs = [block.init_eqs[gate] for gate in block.out_vars[:3]]
    result = np.empty_like(references)
    for row, (t, modulation_values) in enumerate(zip(time, references)):
        sampler.update(t, modulation_values, values)
        bindings.update({var.name: value for var, value in zip(parameters, values)})
        bindings[EmtProblemTemplate.TIME_NAME] = float(t)
        bindings[carrier.name] = carrier_expr.eval(**bindings)
        result[row] = [expr.eval(**bindings) for expr in gate_exprs]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", type=Path, help="Path to the patched variable-dq comparison harness")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("variable_dq_harness", args.harness)
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness
    spec.loader.exec_module(harness)

    trace = harness.simulate_comparison(n_samples=40001)
    references = np.column_stack([trace.ref_a, trace.ref_b, trace.ref_c])
    hard = np.column_stack([trace.proc_gate_a, trace.proc_gate_b, trace.proc_gate_c])
    soft = soft_gates(trace.t, references)
    assert np.isfinite(soft).all() and np.all((soft >= 0) & (soft <= 1))
    time_ms = trace.t * 1000

    fig, axes = plt.subplots(3, 2, figsize=(11, 7), layout="constrained")
    for phase, (overview, detail) in enumerate(axes):
        phase_name = "ABC"[phase]
        # Use an interior gate edge for a close-up of the smoothed transition.
        edge = np.flatnonzero(np.diff(hard[:, phase]))[1] + 1
        for ax in (overview, detail):
            ax.step(time_ms, hard[:, phase], where="post", lw=1.1, label="Procedural hard PWM")
            ax.plot(time_ms, soft[:, phase], "--", lw=1.2, label="Compatible soft PWM")
            ax.set_ylim(-0.08, 1.08)
            ax.grid(alpha=0.23)
        overview.set_ylabel(f"Gate {phase_name}")
        detail.set_xlim(time_ms[edge] - 0.002, time_ms[edge] + 0.002)
        thresholded = soft[:, phase] >= 0.5
        print(f"{phase_name}: hard edges={np.count_nonzero(np.diff(hard[:, phase]))}, "
              f"soft 0.5 crossings={np.count_nonzero(np.diff(thresholded))}, "
              f"sampled mean absolute gate difference={np.mean(np.abs(hard[:, phase] - soft[:, phase])):.6g}")
    axes[0, 0].set_title("Full variable-dq test")
    axes[0, 1].set_title("Individual gate edges")
    axes[0, 0].legend(loc="lower center", ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("Time [ms]")
    output = Path(__file__).resolve().parent / "plots" / "pwm_soft_sign_compare.png"
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == "__main__":
    main()
