"""Run the charged IEEE-9 explicit ODE with native .simulate() and a grid event.

Beginner's map
--------------
1. MODEL: ``extract`` assembles IEEE-9 π lines and two GFL converters, then
   substitutes terminal charge and bus KCL to obtain a 103-state ODE.
2. EVENT: a grid ``EmtEvent`` raises generator 1's active-power reference.
3. SOLVE: one native ``EmtModelProblem.simulate()`` call spans the event.
4. SAVE: ``simulation_data/ieee9_full_explicit`` contains pre/post-event
   S/Phi matrices, named input and state traces, an NPZ, and a plot.

The three original zero-charging lines use B=0.1, as in the charge-reduction
experiment. An EmtEvent changes the first GFL's P-reference scale at 0.05 s.
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
from scipy.io import savemat

from TensyGridEngine.emt.emt_explicit_converter_demo.experiments.ieee9_pi_charge_event_timeseries import (
    balanced_groups, compile_terms, phase_rms, polynomial_rhs,
)
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_pi_charge_native import build_native_problem
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_pi_charge_polynomial import extract
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.simulation_data import case_dir, save_case_data
from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Devices.Events.emt_events_group import EmtEventsGroup
from VeraGridEngine.Simulations.EMT.problems.emt_problem_template import EmtProblemTemplate
from VeraGridEngine.enumerations import EmtSolverTypes


def main(*, event_time=0.05, end_time=0.2, power_step=0.10,
         step=5e-5, solver_type="Symbolic", output_dir=None):
    if not 0 < event_time < end_time or step <= 0 or power_step == 0:
        raise ValueError("Require 0 < event_time < end_time, positive step, nonzero event")
    if not np.isclose(round(event_time / step) * step, event_time, atol=1e-12):
        raise ValueError("Choose a native time step that lands exactly on the event")
    output_dir = Path(output_dir) if output_dir is not None else case_dir("ieee9_full_explicit")
    output_dir.mkdir(parents=True, exist_ok=True)

    # MODEL: assemble and extract both parameter values of the same grid.
    full, before, dynamic, charges, Q, _ = extract(replacement_B=0.1)
    _, after, dynamic_after, _, _, _ = extract(
        replacement_B=0.1, first_gfl_power_scale=1.0 + power_step)
    if not np.array_equal(dynamic, dynamic_after):
        raise RuntimeError("Reduced state ordering changed after the power event")
    n, m = len(full.get_state_vars()), len(full.get_algebraic_vars())
    # The slack Norton source is initialized to supply the PF current, so
    # the raw model initialization can be used directly here.
    x0 = full.get_x0()
    z0 = np.r_[x0[dynamic], x0[n + m - 27:n + m]]
    charge_error = np.linalg.norm(x0[charges] - Q @ z0[len(dynamic):])
    if charge_error > 1e-9:
        raise RuntimeError(f"Initial charge inconsistency: {charge_error:.3e}")

    solver_enum = getattr(EmtSolverTypes, solver_type)
    native = build_native_problem(before, z0, step=step, event_output=after,
                                  event_step=power_step, solver_type=solver_enum,
                                  explicit=True)
    native.options.simulation_time = end_time
    native.grid = full.grid
    first_gfl = next(generator for generator in full.grid.generators
                     if not generator.bus.is_slack)
    nominal_power = first_gfl.P / full.grid.Sbase
    # EVENT: VeraGrid updates the runtime scale in one continuous simulation.
    group = EmtEventsGroup(name="first_gfl_P_reference_step")
    full.grid.add_emt_events_group(group)
    full.grid.add_emt_event(EmtEvent(
        name="first GFL P reference +10%", device=first_gfl,
        parameter=native.event_scale_var, time=event_time,
        value=1.0 + power_step, group=group, force_step_alignment=True,
    ))
    # EmtModelProblem bypasses the network constructor, but the generic EMT
    # event machinery can use its attached grid and matching runtime Var UID.
    EmtProblemTemplate.set_events_group(native, group)
    event_index = native.uid2idx_event_params[native.event_scale_var.uid]
    before_scale = native.def_event_params_fn(native.event_params_values.copy(),
                                              event_time - step)[event_index]
    after_scale = native.def_event_params_fn(native.event_params_values.copy(),
                                             event_time + step)[event_index]
    if not (np.isclose(before_scale, 1.0) and np.isclose(after_scale, 1.0 + power_step)):
        raise RuntimeError(f"Grid event did not update the reduced runtime parameter: {before_scale=}, {after_scale=}")
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

    # SAVE: the event parameter is one during the first segment and 1.1
    # afterwards. Save its evaluated trace alongside the named ODE states.
    scale_trace = np.array([
        native.def_event_params_fn(native.event_params_values.copy(), float(t))[event_index]
        for t in time
    ])
    state_names = [f"{full.get_state_vars()[index].name}_{index}" for index in dynamic]
    state_names += [f"v_{bus.name}_{phase}" for bus in full.grid.buses for phase in "ABC"]
    S, Phi = native.build_multilinear_matrices_fast()
    # One input-aware S/Phi pair must reproduce both event regimes. Check it
    # against the independently extracted coefficient maps at the initial state.
    for scale, coefficients in ((1.0, before), (1.0 + power_step, after)):
        basis = np.r_[z0, scale]
        monomials = np.asarray([
            np.prod(basis[S.indices[S.indptr[column]:S.indptr[column + 1]]])
            for column in range(S.shape[1])
        ])
        expected = polynomial_rhs(z0, *compile_terms(coefficients))
        error = np.max(np.abs(np.asarray(Phi @ monomials).ravel() - expected))
        if error > 1e-7:
            raise RuntimeError(f"Input-aware S/Phi mismatch at scale {scale:g}: {error:.3e}")
    post_model = build_native_problem(after, z0, step=step, solver_type=solver_enum,
                                      explicit=True)
    S_after, Phi_after = post_model.build_multilinear_matrices_fast()
    save_case_data(
        output_dir, S=S, Phi=Phi, time=time, states=state,
        state_names=state_names,
        inputs=np.column_stack((scale_trace, nominal_power * scale_trace)),
        input_names=["first_gfl_power_scale", "gen_1_P_ref_pu"],
        basis_names=[var.name for var in native.get_state_vars()]
                    + [native.event_scale_var.name],
        metadata={"event_time": event_time, "power_step": power_step,
                  "native_step": step, "replacement_line_B": 0.1},
    )
    savemat(output_dir / "S_after.mat", {
        "S": S_after,
        "basis_names": np.asarray([var.name for var in post_model.get_state_vars()],
                                  dtype=object).reshape(1, -1),
    }, do_compression=True)
    savemat(output_dir / "Phi_after.mat", {"Phi": Phi_after}, do_compression=True)
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
    print(f"event scale={before_scale:g}->{after_scale:g} at {event_time:g} s")
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
