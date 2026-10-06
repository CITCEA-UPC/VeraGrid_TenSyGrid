#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Study the abc voltage/current/power base convention used by VeraGrid EMT.

The case is deliberately small: one Thevenin source, one balanced PI line and
one grounded-star resistive load.  It answers two independent questions:

1. Does total per-unit power equal ``sum(v_phase*i_phase)`` or that sum / 3?
2. Does a positive-sequence line ``R/X/B`` map directly to each phase-domain
   equation, or does the selected phase-current base require ``Z*3, Y/3``?

The phasor checks are the decisive part of the study; the EMT run supplies an
independent time-domain power-balance check and plots the relevant waveforms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.EMT.solvers.jit_symbolic_solver import evaluate_batched_residual
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Templates.Emt.load_RLC_emt_template import get_shunt_r_emt_template
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import (
    get_generator_thevenin_rl_emt_template_with_ref,
)
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtSolverTypes,
)

DT = 5.0e-6
SIMULATION_TIME = 0.04


def build_power_flow_options() -> gce.PowerFlowOptions:
    """Return the deterministic Newton options used by this small study."""
    return gce.PowerFlowOptions(
        solver_type=gce.SolverType.NR,
        retry_with_other_methods=False,
        verbose=0,
        tolerance=1.0e-8,
        max_iter=30,
        control_q=False,
        distributed_slack=False,
    )


def build_case() -> tuple[gce.MultiCircuit, gce.Line, gce.Generator, gce.Load]:
    """Build a balanced two-bus source-line-resistor convention test."""
    grid = gce.MultiCircuit(name="EMT per-unit convention study", Sbase=100.0, fbase=50.0)
    source_bus = gce.Bus(name="source bus", Vnom=230.0, is_slack=True)
    load_bus = gce.Bus(name="load bus", Vnom=230.0)
    grid.add_bus(source_bus)
    grid.add_bus(load_bus)

    line = gce.Line(
        name="study line",
        bus_from=source_bus,
        bus_to=load_bus,
        r=0.02,
        x=0.15,
        b=0.12,
        rate=250.0,
    )
    source = gce.Generator(
        name="Thevenin source",
        P=60.0,
        vset=1.0,
        Snom=100.0,
        freq=50.0,
        r1=0.002,
        x1=0.20,
    )
    load = gce.Load(name="resistive load", P=60.0, Q=0.0)
    grid.add_line(line)
    grid.add_generator(bus=source_bus, api_obj=source)
    grid.add_load(bus=load_bus, api_obj=load)

    for bus in grid.buses:
        get_bus_emt_template(grid=grid, bus=bus)
    set_emt_model(
        device=source,
        model=get_generator_thevenin_rl_emt_template_with_ref(
            vf=grid.var_factory, name="study_thevenin"
        ).block,
        var_factory=grid.var_factory,
    )
    set_emt_model(
        device=line,
        model=get_pi_line_emt_template(
            vf=grid.var_factory,
            phN=False,
            phA=True,
            phB=True,
            phC=True,
            name="study_line",
        ).block,
        var_factory=grid.var_factory,
    )
    set_emt_model(
        device=load,
        model=get_shunt_r_emt_template(
            vf=grid.var_factory,
            phA=True,
            phB=True,
            phC=True,
            name="study_resistor",
        ).block,
        var_factory=grid.var_factory,
    )
    return grid, line, source, load


def phasor_convention_metrics(
    grid: gce.MultiCircuit,
    line: gce.Line,
    pf3: Any,
) -> dict[str, float]:
    """Compare PF currents/powers with the two candidate line scalings."""
    f = grid.buses.index(line.bus_from)
    t = grid.buses.index(line.bus_to)
    voltages_f = np.asarray([pf3.voltage_A[f], pf3.voltage_B[f], pf3.voltage_C[f]])
    voltages_t = np.asarray([pf3.voltage_A[t], pf3.voltage_B[t], pf3.voltage_C[t]])
    powers_f = np.asarray([pf3.Sf_A[0], pf3.Sf_B[0], pf3.Sf_C[0]]) / grid.Sbase
    # Sf_A/B/C are physical per-phase powers divided by the *total* three-phase
    # Sbase.  With Vphase_pu based on VLL_base/sqrt(3), conventional current pu
    # therefore contains a factor of three.  Omitting it defines the legacy EMT
    # phase-current base, which is three times larger in amperes.
    currents_legacy = np.conj(powers_f / voltages_f)
    currents_standard = 3.0 * currents_legacy

    z1 = complex(line.R, line.X)
    y1 = 1j * line.B
    currents_direct = (voltages_f - voltages_t) / z1 + 0.5 * y1 * voltages_f
    currents_three = (voltages_f - voltages_t) / (3.0 * z1) + (y1 / 3.0) * 0.5 * voltages_f

    power_legacy_sum = float(np.real(np.sum(voltages_f * np.conj(currents_legacy))))
    power_standard_div3 = float(
        np.real(np.sum(voltages_f * np.conj(currents_standard))) / 3.0
    )
    power_pf = float(np.real(np.sum(powers_f)))
    return {
        "power_pf_pu": power_pf,
        "power_legacy_sum_pu": power_legacy_sum,
        "power_standard_sum_div3_pu": power_standard_div3,
        "standard_current_error_with_z1": float(np.max(np.abs(currents_direct - currents_standard))),
        "legacy_current_error_with_3z1": float(np.max(np.abs(currents_three - currents_legacy))),
        "standard_to_legacy_current_ratio": float(
            np.mean(np.abs(currents_standard / currents_legacy))
        ),
    }


def _three_phase(problem: EmtProblemDae, block: Any, prefix: str, y: np.ndarray) -> np.ndarray:
    values: list[np.ndarray] = []
    for phase in "ABC":
        variable = find_name_in_block(f"{prefix}_{phase}", block)
        if variable is None:
            raise RuntimeError(f"Missing {prefix}_{phase} in {block.name}")
        values.append(y[:, int(problem.get_var_idx(variable))])
    return np.column_stack(values)


def cycle_average(t: np.ndarray, values: np.ndarray, frequency: float) -> float:
    """Average a signal over the final complete fundamental cycle."""
    period = 1.0 / frequency
    mask = t >= (t[-1] - period)
    return float(np.trapz(values[mask], t[mask]) / (t[mask][-1] - t[mask][0]))


def run_emt(
    grid: gce.MultiCircuit,
    pf: Any,
    pf3: Any,
) -> tuple[EmtProblemDae, np.ndarray, np.ndarray, bool, bool]:
    options = EmtOptions(
        time_step=DT,
        simulation_time=SIMULATION_TIME,
        tolerance=1.0e-6,
        solver_type=EmtSolverTypes.Symbolic,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        init_dense_threshold=10_000,
        verbose=0,
    )
    problem = EmtProblemDae(
        grid=grid,
        options=options,
        pf_results_3ph=pf3,
        pf_results=pf,
    )
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=SIMULATION_TIME,
        h=DT,
        method=options.integration_method,
    )
    t, y, _dy, initialized, converged = solver.simulate(boundary_updater=cast(Any, problem))
    return problem, t, y, bool(initialized), bool(converged)


def first_step_verification(time_step: float) -> tuple[float, float, float]:
    """Measure the initial residual and line-current projection for one step."""
    grid, line, _source, _load = build_case()
    pf_options = build_power_flow_options()
    balanced = PowerFlowDriver(grid=grid, options=pf_options)
    balanced.run()
    three_phase = PowerFlowDriver3Ph(grid=grid, options=pf_options)
    three_phase.run()
    options = EmtOptions(
        time_step=time_step,
        simulation_time=time_step,
        tolerance=1.0e-8,
        solver_type=EmtSolverTypes.Symbolic,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        init_dense_threshold=10_000,
        verbose=0,
    )
    problem = EmtProblemDae(
        grid=grid,
        options=options,
        pf_results_3ph=three_phase.results,
        pf_results=balanced.results,
    )
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=time_step,
        h=time_step,
        method=options.integration_method,
    )
    solver.build_jit_kernel(options.integration_method)
    x0, dx0 = problem.get_x0(), problem.get_dx0()
    event_values = problem.def_event_params_fn(problem.event_params_values.copy(), 0.0)
    full_parameters = np.r_[event_values, [float(p.value) for p in problem.get_parameters_values()]]
    problem.update(0.0, x0, full_parameters)
    residual = np.zeros_like(x0)
    evaluate_batched_residual(
        solver.jit_kernels[options.integration_method],
        x0, full_parameters, x0, dx0, time_step, x0, residual,
    )
    max_algebraic_residual = float(
        np.max(np.abs(residual[problem.get_states_number():]))
    )

    _t, y, _dy, _initialized, _converged = solver.simulate(boundary_updater=problem)
    omega = 2.0 * np.pi * grid.fBase
    errors: list[float] = []
    jumps: list[float] = []
    voltage_arrays = (
        three_phase.results.voltage_A,
        three_phase.results.voltage_B,
        three_phase.results.voltage_C,
    )
    terminal_data = (
        ("if", line.bus_from, (three_phase.results.Sf_A, three_phase.results.Sf_B, three_phase.results.Sf_C)),
        ("it", line.bus_to, (three_phase.results.St_A, three_phase.results.St_B, three_phase.results.St_C)),
    )
    for prefix, bus, power_arrays in terminal_data:
        bus_index = grid.buses.index(bus)
        for phase, powers, voltages in zip("ABC", power_arrays, voltage_arrays):
            variable = find_name_in_block(f"{prefix}_{phase}", line.emt_model)
            if variable is None:
                raise RuntimeError(f"Missing line variable {prefix}_{phase}")
            index = int(problem.get_var_idx(variable))
            phasor = np.conj((powers[0] / grid.Sbase) / voltages[bus_index])
            expected = np.sqrt(2.0) * np.imag(
                phasor * np.exp(1j * omega * time_step)
            )
            errors.append(abs(float(y[1, index]) - expected))
            jumps.append(abs(float(y[1, index]) - float(y[0, index])))
    return max_algebraic_residual, max(errors), max(jumps)


def save_plot(problem: EmtProblemDae, t: np.ndarray, y: np.ndarray) -> Path:
    source, load = problem.grid.generators[0], problem.grid.loads[0]
    source_bus, load_bus = problem.grid.buses
    v_source = _three_phase(problem, source_bus.emt_model, "v", y)
    v_load = _three_phase(problem, load_bus.emt_model, "v", y)
    i_source = _three_phase(problem, source.emt_model, "i", y)
    i_load = _three_phase(problem, load.emt_model, "i", y)
    p_source_sum = np.sum(v_source * i_source, axis=1)
    p_load_sum = -np.sum(v_load * i_load, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True, constrained_layout=True)
    axes[0].plot(t * 1e3, v_load)
    axes[0].set_ylabel("Load voltage [pu]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t * 1e3, i_source)
    axes[1].set_ylabel("Source current [pu]")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(t * 1e3, p_source_sum, label="source: Σvi")
    axes[2].plot(t * 1e3, p_source_sum / 3.0, label="source: Σvi/3")
    axes[2].plot(t * 1e3, p_load_sum, label="load: −Σvi", alpha=0.8)
    axes[2].set_ylabel("Instantaneous power [pu]")
    axes[2].set_xlabel("Time [ms]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=3)
    fig.suptitle("EMT per-unit convention study")
    output = Path(__file__).with_name("emt_per_unit_convention_study.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print("emt_last_cycle_source_sum_pu=", cycle_average(t, p_source_sum, problem.grid.fBase))
    print("emt_last_cycle_source_sum_div3_pu=", cycle_average(t, p_source_sum / 3.0, problem.grid.fBase))
    print("emt_last_cycle_load_sum_pu=", cycle_average(t, p_load_sum, problem.grid.fBase))
    return output


def main() -> None:
    grid, line, _source, _load = build_case()
    pf_options = build_power_flow_options()
    balanced = PowerFlowDriver(grid=grid, options=pf_options)
    balanced.run()
    three_phase = PowerFlowDriver3Ph(grid=grid, options=pf_options)
    three_phase.run()
    if not bool(balanced.results.converged) or not bool(three_phase.results.converged):
        raise RuntimeError("Power flow did not converge")

    metrics = phasor_convention_metrics(grid, line, three_phase.results)
    print("=== PF phasor convention checks ===")
    for name, value in metrics.items():
        print(f"{name}={value:.12g}")
    print("conventional_contract= I=3*conj(Sphase_pu/Vphase_pu), P=sum(vi)/3, Zphase=Z1")
    print("legacy_contract= I=conj(Sphase_pu/Vphase_pu), P=sum(vi), Zphase=3*Z1")

    problem, t, y, initialized, converged = run_emt(grid, balanced.results, three_phase.results)
    print(f"emt_initialized={initialized}")
    print(f"emt_converged={converged}")
    print(f"emt_finite={bool(np.isfinite(y).all())}")
    print("plot=", save_plot(problem, t, y))

    print("=== First-step convergence check (current implementation) ===")
    print("h_us,max_algebraic_residual,max_line_current_error,max_line_current_jump")
    for time_step in (20e-6, 10e-6, 5e-6, 2.5e-6, 1.25e-6):
        algebraic_residual, current_error, current_jump = first_step_verification(time_step)
        print(
            f"{time_step * 1e6:.3f},{algebraic_residual:.12g},"
            f"{current_error:.12g},{current_jump:.12g}"
        )


if __name__ == "__main__":
    main()
