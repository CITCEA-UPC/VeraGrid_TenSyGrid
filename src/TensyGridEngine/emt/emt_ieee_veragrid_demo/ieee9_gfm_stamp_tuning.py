#!/usr/bin/env python3
"""Compare the existing IEEE9 UPC GFM gains with a STAMP-like tuning.

Only parameters are changed; the VeraGrid UPC equations, Park convention,
decoupling terms, initialization, and network are identical in both cases.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt import ieee9_emt_ibr_simulation as ieee9
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.enumerations import DynamicIntegrationMethod


@dataclass(frozen=True)
class Tuning:
    kp_v: float
    ki_v: float
    kp_i: float
    ki_i: float
    kdp: float
    kdq: float
    tau_p: float
    tau_q: float


BASELINE = Tuning(0.00075, 0.2, 0.00075, 0.2, 0.003, 0.005, 0.01, 0.01)
TEMPLATE_DEFAULT = Tuning(0.1, 0.5, 1.0, 5.0, 0.003, 0.005, 0.01, 0.01)
FAST_CURRENT_ONLY = Tuning(0.1, 0.5, 0.47746482927568595, 20.0,
                           0.003, 0.005, 0.01, 0.01)
STAMP_DROOP_ONLY = Tuning(0.1, 0.5, 1.0, 5.0, 0.05, 1.0/15.0, 0.1, 0.1)


def stamp_like_tuning(frequency_hz: float = 50.0) -> Tuning:
    """Map STAMP's 1-ms current and 50-ms voltage design to UPC units."""
    omega_base = 2.0 * np.pi * frequency_hz
    current_settling = 1.0e-3
    voltage_settling = 0.05
    damping = 0.707
    lf, rf, cf = 0.15, 0.02, 0.05
    natural_frequency = 4.0 / (voltage_settling * damping)
    return Tuning(
        kp_v=2.0 * damping * natural_frequency * (cf / omega_base),
        ki_v=natural_frequency**2 * (cf / omega_base),
        kp_i=(lf / omega_base) / current_settling,
        ki_i=rf / current_settling,
        kdp=0.05,
        kdq=1.0 / 15.0,
        tau_p=0.1,
        tau_q=0.1,
    )


def build_case(tuning: Tuning, simulation_time: float, time_step: float):
    os.environ["VERAGRID_GFM_EMT_KP_VCL"] = str(tuning.kp_v)
    os.environ["VERAGRID_GFM_EMT_KI_VCL"] = str(tuning.ki_v)
    os.environ["VERAGRID_GFM_EMT_KP_ICL"] = str(tuning.kp_i)
    os.environ["VERAGRID_GFM_EMT_KI_ICL"] = str(tuning.ki_i)
    grid, _gfl, _gfl_buses, gfm_buses, gfm_blocks, validation = ieee9.build_ibr_grid(
        n_gfl=0, n_gfm=1, multilinear_inverters=False)
    model = gfm_blocks[0][1]
    for generator in grid.generators:
        if generator.bus.name not in gfm_buses:
            generator.emt_model.set_parameter_in_model("K", 40.0)
    vf = grid.var_factory
    for name, value in (
        ("Kdp", tuning.kdp), ("Kdq", tuning.kdq),
        ("tau_P", tuning.tau_p), ("tau_Q", tuning.tau_q),
    ):
        variable = validation.find_name_in_block(name, model)
        if variable is None or variable not in model.event_dict:
            raise KeyError(f"GFM event parameter {name!r} was not found")
        model.event_dict[variable] = vf.add_const(value)

    pf = PowerFlowDriver(grid=grid, options=ieee9.build_power_flow_options()); pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=ieee9.build_power_flow_options()); pf3.run()
    if not pf.results.converged or not pf3.results.converged:
        raise RuntimeError("IEEE9 power flow failed")
    options = ieee9.build_emt_options()
    options.time_step = time_step
    options.simulation_time = simulation_time
    options.integration_method = DynamicIntegrationMethod.DaeTrapezoidal
    options.tolerance = 1.0e-9
    problem = EmtProblemDae(grid=grid, options=options,
                            pf_results=pf.results, pf_results_3ph=pf3.results)
    validation.seed_bus_algebraic_predictors(problem, grid, pf.results)
    return grid, model, validation, problem, pf3.results


def trace(problem, values, model, name):
    value = ieee9._get_signal(problem, values, model, name)
    if value is None:
        raise KeyError(name)
    return np.asarray(value, dtype=float)


def simulate(label: str, tuning: Tuning, duration: float, step: float):
    grid, model, validation, problem, pf3 = build_case(tuning, duration, step)
    time, values, _derivatives, initialized, converged = validation.run_emt_with_p_ref_step(
        problem, problem.options, model, step_time=0.02, step_delta=1.0e-3)
    signals = {name: trace(problem, values, model, name) for name in ("omega", "P", "Q")}
    print(f"{label}: initialized={initialized}, converged={converged}")
    for name, values_ in signals.items():
        pre = float(np.mean(values_[time < 0.02]))
        print(f"  {name}: peak deviation={np.max(np.abs(values_[time >= 0.02]-pre)):.6e}")
    return grid, model, problem, pf3, np.asarray(time), signals


def run_ssa(grid, problem, pf3, modes: int):
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import SmallSignalStabilityEmtDriver
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import SmallSignalStabilityEmtOptions
    options = SmallSignalStabilityEmtOptions(
        k=modes, target_period=1.0/grid.fBase, ss_assessment_time=0.06, verbose=1)
    driver = SmallSignalStabilityEmtDriver(
        grid=grid, emt_options=problem.options, sss_options=options, pf_results=pf3)
    driver.problem = problem
    driver.run()
    return driver.results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.08)
    parser.add_argument("--time-step", type=float, default=20e-6)
    parser.add_argument("--ssa", action="store_true")
    parser.add_argument(
        "--profiles", default="weak,stamp",
        help="comma-separated: weak, default, fast-current, stamp-droop, stamp",
    )
    args = parser.parse_args()
    available = {
        "weak": ("existing weak tuning", BASELINE),
        "default": ("template default tuning", TEMPLATE_DEFAULT),
        "fast-current": ("fast current loop only", FAST_CURRENT_ONLY),
        "stamp-droop": ("STAMP droop/filter only", STAMP_DROOP_ONLY),
        "stamp": ("STAMP-like tuning", stamp_like_tuning()),
    }
    profiles = [available[name.strip()] for name in args.profiles.split(",")]
    runs = []
    for label, tuning in profiles:
        print(f"{label}: {tuning}")
        runs.append((label, *simulate(label, tuning, args.duration, args.time_step)))

    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for label, _grid, _model, _problem, _pf3, time, signals in runs:
        for axis, name in zip(axes, ("omega", "P", "Q")):
            axis.plot(1e3*time, signals[name]-signals[name][0], label=label)
    for axis, name in zip(axes, ("omega", "P", "Q")):
        axis.set_ylabel(f"delta {name} (pu)")
        axis.axvline(20.0, color="black", linestyle=":")
        axis.grid(True, alpha=0.3); axis.legend()
    axes[-1].set_xlabel("time (ms)")
    axes[0].set_title("IEEE9 UPC GFM: existing versus STAMP-like tuning")
    output = Path(__file__).with_name("ieee9_gfm_stamp_tuning.png")
    figure.savefig(output, dpi=180); plt.close(figure)
    print(f"plot={output}")

    if args.ssa:
        for label, grid, _model, problem, pf3, _time, _signals in runs:
            result = run_ssa(grid, problem, pf3, modes=24)
            magnitudes = np.abs(result.multipliers)
            unstable = magnitudes > 1.0 + 1.0e-6
            print(f"{label} SSA: max|mu|={magnitudes.max():.9e}, unstable={unstable.sum()}")
            for index in np.argsort(magnitudes)[::-1][:8]:
                print(f"  mu={result.multipliers[index]:+.7e}, lambda={result.eigenvalues[index]:+.7e}")
            dominant = int(np.argmax(magnitudes))
            participants = np.argsort(result.participation_factors[:, dominant])[::-1][:10]
            print("  dominant participants:")
            for state_index in participants:
                print(f"    {result.participation_factors[state_index, dominant]:.4e} "
                      f"{result.stat_vars_array[state_index]}")


if __name__ == "__main__":
    main()
