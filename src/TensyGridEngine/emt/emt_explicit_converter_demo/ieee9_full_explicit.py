"""Run the charged IEEE-9 explicit ODE with native .simulate() and a grid event.

Beginner's map
--------------
1. MODEL: this script calls ``extract`` in ``utils/ieee9_pi_charge_polynomial.py``.
   That helper calls ``build_problem`` in ``utils/ieee9_gfl_ode_reduced.py``
   to assemble the IEEE-9 device and network equations. ``extract`` then
   eliminates charge constraints and network algebraics to obtain a 103-state
   polynomial ODE. ``build_native_problem`` in ``utils/ieee9_pi_charge_native.py``
   turns its coefficient maps into the Block equations that we simulate.
2. EVENT: a grid ``EmtEvent`` raises generator 1's active-power reference.
3. SOLVE: one native ``EmtModelProblem.simulate()`` call spans the event.
4. SAVE: ``simulation_data/ieee9_full_explicit`` contains one input-aware
   S/Phi pair, named input and state traces, an NPZ, and a plot.

The three original zero-charging lines use B=0.1, as in the charge-reduction
experiment. An EmtEvent changes the first GFL's P_ref directly at 0.05 s.
The complete 103-coordinate ODE is integrated without phase projection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# A direct `python path/to/this_script.py` launch does not add repository/src.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt.emt_explicit_converter_demo.experiments.ieee9_pi_charge_event_timeseries import (
    balanced_groups, compile_terms, phase_rms, polynomial_rhs,
)
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_pi_charge_native import build_native_problem
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_pi_charge_polynomial import extract, specialize_reference
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.explicit_input_events import ExplicitInputEvents
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.simulation_data import case_dir, save_case_data
from VeraGridEngine.enumerations import EmtSolverTypes


def main(*, event_time=0.05, end_time=0.2, power_step=0.10,
         step=5e-5, solver_type="Symbolic", output_dir=None):
    if not 0 < event_time < end_time or step <= 0 or power_step == 0:
        raise ValueError("Require 0 < event_time < end_time, positive step, nonzero event")
    if not np.isclose(round(event_time / step) * step, event_time, atol=1e-12):
        raise ValueError("Choose a native time step that lands exactly on the event")
    output_dir = Path(output_dir) if output_dir is not None else case_dir("ieee9_full_explicit")
    output_dir.mkdir(parents=True, exist_ok=True)

    # MODEL CALL CHAIN: extract() calls build_problem() to assemble the original
    # IEEE-9 EMT DAE from π-line, converter, and source templates. It then
    # eliminates terminal-charge constraints and network algebraic variables.
    # `output` contains all 103 RHS polynomials. Its final mode is the first
    # GFL's actual P_ref, carried symbolically through this ONE grid reduction.
    events = ExplicitInputEvents("first_gfl_P_reference_step")
    # Register event-controlled references before extraction. This makes each
    # reference a symbolic input mode in the reduced ODE and in S/Phi.
    nominal_power = 1.63  # IEEE-9 gen 1 active power, pu on the grid base.
    stepped_power = nominal_power * (1.0 + power_step)
    events.add_event_to_inputs(
        generator="gen 1", parameter="P_ref", time=event_time,
        value=stepped_power, name="first GFL P reference +10%")
    full, output, dynamic, charges, Q, _ = extract(
        replacement_B=0.1, symbolic_references=events.references)
    first_gfl = next(generator for generator in full.grid.generators
                     if not generator.bus.is_slack)
    nominal_power = first_gfl.P / full.grid.Sbase
    stepped_power = nominal_power * (1.0 + power_step)
    # The initialized grid parameter must match the generator operating point.
    if not np.isclose(events.initial_values(full)[0], nominal_power):
        raise RuntimeError("Extracted P_ref differs from the grid operating point")
    n, m = len(full.get_state_vars()), len(full.get_algebraic_vars())
    # The slack Norton source is initialized to supply the PF current, so
    # the raw model initialization can be used directly here.
    x0 = full.get_x0()
    z0 = np.r_[x0[dynamic], x0[n + m - 27:n + m]]
    charge_error = np.linalg.norm(x0[charges] - Q @ z0[len(dynamic):])
    if charge_error > 1e-9:
        raise RuntimeError(f"Initial charge inconsistency: {charge_error:.3e}")

    solver_enum = getattr(EmtSolverTypes, solver_type)
    # This is where the reduced polynomial becomes executable equations:
    # build_native_problem creates z_0,...,z_102 and one symbolic RHS per z_i,
    # stores those expressions in Block.state_eqs, and wraps the Block in an
    # EmtModelProblem. `.simulate()` below integrates that Block, not the
    # original grid DAE returned by extract().
    input_names, input_values = events.input_names, events.initial_values(full)
    native = build_native_problem(output, z0, step=step,
                                  input_names=input_names, input_values=input_values,
                                  solver_type=solver_enum,
                                  explicit=True)
    native.options.simulation_time = end_time
    # EVENT: the helper binds the declared grid events to the reduced inputs.
    events.install(full, native)
    reference_var = native.input_vars_by_name[input_names[0]]
    event_index = native.uid2idx_event_params[reference_var.uid]
    before_ref = native.def_event_params_fn(native.event_params_values.copy(),
                                            event_time - step)[event_index]
    after_ref = native.def_event_params_fn(native.event_params_values.copy(),
                                           event_time + step)[event_index]
    if not (np.isclose(before_ref, nominal_power) and np.isclose(after_ref, stepped_power)):
        raise RuntimeError(f"Grid event did not update P_ref: {before_ref=}, {after_ref=}")
    # SOLVE: native symbolic EMT integration uses the initialized ODE slopes.
    print(f"starting native .simulate(): step={step:g} s, samples≈{end_time / step:.0f}", flush=True)
    result = native.simulate(t0=0.0, t_end=end_time)
    if not result.initialized or not result.converged:
        raise RuntimeError(f"Native EMT simulation failed: {result.initialized=}, {result.converged=}")
    time, state = result.time, result.values
    voltage = state[:, len(dynamic):].reshape(-1, 9, 3)
    source_names = [variable.name for variable in full.get_state_vars()]
    current_indices = []
    for generator in ("gen 1", "gen 2"):
        indices = [next(i for i, variable in enumerate(source_names)
                        if variable == f"i_network_{generator}_{phase}") for phase in "ABC"]
        current_indices.append([int(np.flatnonzero(dynamic == index)[0]) for index in indices])
    voltage_rms = np.column_stack([phase_rms(voltage[:, bus]) for bus in (1, 2, 5)])
    current_rms = np.column_stack([phase_rms(state[:, indices]) for indices in current_indices])
    active_power = np.column_stack([
        np.sum(voltage[:, bus] * state[:, indices], axis=1) / 3.0
        for bus, indices in zip((1, 2), current_indices)
    ])
    groups = balanced_groups(full, dynamic)
    max_zero_sequence = max(float(np.max(abs(np.mean(state[:, indices], axis=1))))
                            for indices in groups)

    # Save the actual P_ref trace and its convenient dimensionless scale.
    reference_trace = np.array([
        native.def_event_params_fn(native.event_params_values.copy(), float(t))[event_index]
        for t in time
    ])
    scale_trace = reference_trace / nominal_power
    state_names = [f"{full.get_state_vars()[index].name}_{index}" for index in dynamic]
    state_names += [f"v_{bus.name}_{phase}" for bus in full.grid.buses for phase in "ABC"]
    S, Phi = native.build_multilinear_matrices_fast()
    # Check the same input-aware S/Phi at both reference values. The event
    # changes an input value, not the equations or their matrices.
    for reference_value in (nominal_power, stepped_power):
        basis = np.r_[z0, reference_value]
        monomials = np.asarray([
            np.prod(basis[S.indices[S.indptr[column]:S.indptr[column + 1]]])
            for column in range(S.shape[1])
        ])
        fixed = specialize_reference(output, reference_value)
        expected = polynomial_rhs(z0, *compile_terms(fixed))
        error = np.max(np.abs(np.asarray(Phi @ monomials).ravel() - expected))
        if error > 1e-7:
            raise RuntimeError(f"Input-aware S/Phi mismatch at P_ref {reference_value:g}: {error:.3e}")
    save_case_data(
        output_dir, S=S, Phi=Phi, time=time, states=state,
        state_names=state_names,
        inputs=np.column_stack((scale_trace, reference_trace)),
        input_names=["first_gfl_power_scale", "gen_1_P_ref_pu"],
        basis_names=[var.name for var in native.get_state_vars()]
                    + [var.name for var in native.sys_block.in_vars],
        metadata={"event_time": event_time, "power_step": power_step,
                  "native_step": step, "replacement_line_B": 0.1},
    )
    data_path = output_dir / "timeseries.npz"
    plot_path = output_dir / "comparison.png"
    np.savez_compressed(data_path, time=time, state=state,
                        bus_voltage_rms=voltage_rms, converter_current_rms=current_rms,
                        converter_active_power=active_power, event_time=event_time,
                        power_step=power_step, native_step=step)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for j, bus in enumerate((1, 2, 5)):
        axes[0].plot(time, voltage_rms[:, j], label=f"bus {bus}")
    for j, generator in enumerate(("gen 1", "gen 2")):
        axes[1].plot(time, current_rms[:, j], label=generator)
        axes[2].plot(time, active_power[:, j], label=f"{generator} measured")
    axes[2].step(time, nominal_power * scale_trace,
                 where="post", ls="--", label="gen 1 reference")
    for axis, ylabel in zip(axes, ("Voltage RMS (pu)", "GFL current RMS (pu)",
                                    "Generated active power (pu)")):
        axis.axvline(event_time, color="black", ls="--", lw=1)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Modified IEEE-9 EMT: native .simulate() with GFL power event")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    event_sample = np.searchsorted(time, event_time)
    print(f"native samples={len(time)}, final time={time[-1]:.6g} s")
    print(f"event P_ref={before_ref:g}->{after_ref:g} pu at {event_time:g} s")
    print(f"gen 1 power pre/final={active_power[event_sample - 1, 0]:.6g}/{active_power[-1, 0]:.6g} pu")
    print(f"maximum ABC mean={max_zero_sequence:.3e}")
    print(f"plot={plot_path}\ndata={data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-time", type=float, default=0.05)
    parser.add_argument("--end-time", type=float, default=0.2)
    parser.add_argument("--power-step", type=float, default=0.10)
    parser.add_argument("--step", type=float, default=5e-5)
    parser.add_argument("--solver-type", choices=[member.name for member in EmtSolverTypes],
                        default="Symbolic")
    parser.add_argument("--output-dir", type=Path)
    main(**vars(parser.parse_args()))
