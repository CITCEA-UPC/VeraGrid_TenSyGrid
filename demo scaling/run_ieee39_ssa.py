#!/usr/bin/env python3
"""Exploratory IEEE-39 RMS and EMT SSA on the stable RMS operating point.

RMS uses the models saved in hackaton_files/grids/ieee39_rms.veragrid.
EMT attaches the existing simple-generator, pi-line, and constant-power ZIP
templates to a fresh load of that same network and dispatch. These are not
equivalent generator models; separate stability results must be reported.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "src/TensyGridEngine/emt")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/veragrid-ieee39-mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/veragrid-ieee39-numba")
os.environ.setdefault("VERAGRID_EMT_INIT_CACHE_DIR", "/tmp/veragrid-ieee39-emt-init")
os.environ.setdefault("VERAGRID_JIT_CACHE_DIR", "/tmp/veragrid-ieee39-jit")
os.environ.setdefault("VERAGRID_RUNTIME_PARAMETER_PYTHON", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from VeraGridEngine.api import open_file
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import SmallSignalStabilityRmsDriver
from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_options import RmsSmallSignalStabilityOptions

GRID = ROOT / "hackaton_files/grids/ieee39_rms.veragrid"
STATIC_GRID = ROOT / "Grids_and_profiles/grids/IEEE39.gridcal"
OUT = Path(__file__).resolve().parent


def save(name: str, values: np.ndarray, elapsed: float, multipliers: bool = False) -> None:
    values = np.asarray(values, dtype=complex)
    with (OUT / f"ieee39_{name}_modes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("mode", "real", "imag", "magnitude"))
        for i, z in enumerate(values):
            writer.writerow((i, z.real, z.imag, abs(z)))
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.scatter(values.real, values.imag, s=25)
    if multipliers:
        angle = np.linspace(0, 2*np.pi, 500)
        ax.plot(np.cos(angle), np.sin(angle), "k--", lw=1)
        ax.set(xlabel="Re(mu)", ylabel="Im(mu)", title="IEEE-39 EMT Floquet multipliers")
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set(xlabel="Re(lambda) [1/s]", ylabel="Im(lambda) [rad/s]",
               title="IEEE-39 RMS eigenvalues")
    ax.grid(alpha=.3)
    fig.savefig(OUT / f"ieee39_{name}_ssa.png", dpi=180)
    plt.close(fig)
    print(f"{name}: modes={len(values)}, elapsed={elapsed:.2f}s, plot={OUT / f'ieee39_{name}_ssa.png'}", flush=True)


def rms() -> None:
    start = perf_counter()
    grid = open_file(str(GRID))
    pf = PowerFlowDriver(grid); pf.run()
    if not pf.results.converged:
        raise RuntimeError("RMS power flow did not converge")
    driver = SmallSignalStabilityRmsDriver(
        grid=grid, pf_results=pf.results,
        sss_options=RmsSmallSignalStabilityOptions(ss_assessment_time=0.0))
    driver.run()
    values = np.asarray(driver.results.eigenvalues)
    physical = values[np.isfinite(values) & (np.abs(values) > 1e-8)]
    print(f"RMS max physical real part={np.max(physical.real):.6g} 1/s", flush=True)
    save("rms", values, perf_counter()-start)


def static_grid_with_rms_dispatch():
    grid = open_file(str(STATIC_GRID))
    reference = open_file(str(GRID))
    for group, attributes in (("generators", ("P", "Vset")),
                              ("loads", ("P", "Q"))):
        source = {device.idtag: device for device in getattr(reference, group)}
        for device in getattr(grid, group):
            saved = source[device.idtag]
            for attribute in attributes:
                setattr(device, attribute, getattr(saved, attribute))
    return grid


def audit_emt_initialization(problem, options) -> None:
    """Check original continuous equations at four phases before running SSA."""
    from copy import deepcopy
    from VeraGridEngine.Simulations.EMT.initialization_emt import (
        validate_emt_periodic_initialization, _emt_initialization_bindings)
    from VeraGridEngine.enumerations import EmtInitializationStatus

    report = validate_emt_periodic_initialization(
        problem, problem._device_models_for_ptc, options,
        deepcopy(problem.initialization_report))
    if report.status != EmtInitializationStatus.RESOLVED:
        raise RuntimeError(f"Initial periodic DAE validation failed: {report.message}")
    bindings = _emt_initialization_bindings(problem)
    rows = [(i, float(eq.eval_uid(bindings)), str(eq))
            for i, eq in enumerate(problem.get_algebraic_eqs())]
    rows.sort(key=lambda row: -abs(row[1]))
    with (OUT / "ieee39_emt_initial_residuals.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("algebraic_row", "residual", "equation"))
        writer.writerows(rows)
    print(f"Original periodic DAE residual at four phases={report.final_residual_inf:.6g}",
          flush=True)


def emt(modes: int, steps: int, assessment_periods: int = 1, periodic_samples: int = 0,
        periodic_method: str = "newton", stability_profile: bool = False) -> None:
    from TensyGridEngine.emt.ieee9_emt_simulation import attach_emt_models
    from TensyGridEngine.emt.emt_power_flow import run_emt_power_flows
    from TensyGridEngine.emt.ieee39_emt_simulation import (
        build_power_flow_options, build_emt_options, configure_ieee39_simple_machines)
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import SmallSignalStabilityEmtDriver
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import SmallSignalStabilityEmtOptions
    from VeraGridEngine.enumerations import SmallSignalEmtBuildTypes, EmtSolverTypes, EmtInitializationMethod, LogSeverity

    start = perf_counter()
    grid = static_grid_with_rms_dispatch()
    attach_emt_models(grid)
    configure_ieee39_simple_machines(grid)
    print("IEEE39 EMT: using benchmark-based single-field machine parameters", flush=True)
    print("IEEE39 EMT: computing operating point", flush=True)
    balanced, three_phase = run_emt_power_flows(grid, build_power_flow_options())
    if stability_profile:
        from TensyGridEngine.emt.ieee39_emt_simulation import configure_ieee39_stability_profile
        configure_ieee39_stability_profile(grid, balanced.voltage)
        print("Experimental profile: 25% Z / 75% P, D/(2H)=0.25/s, excitation gain=2", flush=True)
    print(f"Operating point finished in {perf_counter()-start:.2f}s", flush=True)
    period = 1.0 / grid.fBase
    options = build_emt_options()
    options.solver_type = EmtSolverTypes.StructuralAD
    options.init_newton_max_iter = 50
    if periodic_samples:
        options.initialization_method = (EmtInitializationMethod.PseudoTransient
                                         if periodic_method == "pseudo-transient"
                                         else EmtInitializationMethod.ConsistentNewton)
        options.init_periodic_samples = periodic_samples
    options.time_step = period / steps
    options.simulation_time = assessment_periods * period
    ss_options = SmallSignalStabilityEmtOptions(
        k=modes, target_period=period, ss_assessment_time=options.simulation_time,
        max_krylov_dim=max(30, 2*modes), max_restarts=0, verbose=1,
        consistent_dae_tangent=True,
        build_type=SmallSignalEmtBuildTypes.Arnoldi)
    build_start = perf_counter()
    print("IEEE39 EMT: building DAE problem", flush=True)
    driver = SmallSignalStabilityEmtDriver(grid=grid, emt_options=options,
                                           sss_options=ss_options, pf_results=balanced,
                                           pf_results_3ph=three_phase)
    problem = driver.problem
    print(f"DAE built in {perf_counter()-build_start:.2f}s: "
          f"states={problem.get_states_number()}, algebraics={problem.get_algebraic_var_number()}, "
          f"frequency={grid.fBase:g}Hz, steps/period={steps}, "
          f"assessment_periods={assessment_periods}", flush=True)
    audit_emt_initialization(problem, options)
    ssa_start = perf_counter()
    driver.run()
    print(f"EMT SSA driver elapsed={perf_counter()-ssa_start:.2f}s", flush=True)
    if driver.results is None:
        for entry in driver.logger.entries:
            if entry.severity != LogSeverity.Information:
                print(entry, flush=True)
        raise RuntimeError("EMT SSA returned no result")
    values = np.asarray(driver.results.multipliers)
    trajectory = driver.last_period_trajectory
    if trajectory is not None and len(trajectory) > 1:
        from VeraGridEngine.Simulations.EMT.initialization_emt import _emt_is_absolute_angle
        closure = trajectory[-1] - trajectory[0]
        for var in problem.get_state_vars():
            if _emt_is_absolute_angle(var):
                closure[problem.get_var_idx(var)] -= 2 * np.pi
        scales = np.maximum(1.0, np.max(np.abs(trajectory), axis=0))
        print(f"Last-period max scaled closure error={np.max(np.abs(closure)/scales):.6g}", flush=True)
        np.savez_compressed(OUT / "ieee39_emt_capture.npz", trajectory=trajectory,
                            derivatives=driver.last_period_derivative_trajectory,
                            multipliers=values, period=period, step=options.time_step,
                            scaled_closure=np.max(np.abs(closure)/scales))
    print(f"EMT max |mu|={np.max(np.abs(values)):.9g}", flush=True)
    save("emt", values, perf_counter()-start, multipliers=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("rms", "emt"))
    parser.add_argument("--emt-modes", type=int, default=12)
    parser.add_argument("--steps-per-period", type=int, default=600)
    parser.add_argument("--assessment-periods", type=int, default=1)
    parser.add_argument("--periodic-samples", type=int, default=0,
                        help="Use native periodic initialization with an odd sample count >= 9")
    parser.add_argument("--periodic-method", choices=("newton", "pseudo-transient"), default="newton")
    parser.add_argument("--stability-profile", action="store_true",
                        help="Apply the experimental ZIP, damping and voltage-regulator profile")
    args = parser.parse_args()
    if args.steps_per_period <= 0 or args.assessment_periods <= 0 or args.emt_modes <= 0:
        parser.error("steps, assessment periods and mode count must be positive")
    if args.periodic_samples and (args.periodic_samples < 9 or args.periodic_samples % 2 == 0):
        parser.error("periodic samples must be zero or an odd number >= 9")
    if not GRID.exists():
        raise FileNotFoundError(GRID)
    if args.only != "emt":
        rms()
    if args.only != "rms":
        emt(args.emt_modes, args.steps_per_period, args.assessment_periods,
            args.periodic_samples, args.periodic_method, args.stability_profile)


if __name__ == "__main__":
    main()
