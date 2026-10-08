#!/usr/bin/env python3
"""Compare full nonlinear and linear-deviation STAMP converters on IEEE-9 EMT.

The full nonlinear GFOR/GFOL blocks come from this repository. STAMP_Public is
used only for the shared IEEE-9 case, synchronous generator, and reference
linear-deviation converters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt.emt_stamp_demo.stamp_common import add_stamp_public_to_path
from TensyGridEngine.emt.stamp_converters import (
    STAMP_GFOL,
    STAMP_GFOR,
    build_stamp_nonlinear_converter_emt,
)

HERE = Path(__file__).resolve().parent


def build_grid(full_nonlinear: bool):
    add_stamp_public_to_path()
    from veragrid_stamp.emt_case import build_stamp_wscc_emt_grid
    from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model

    grid = build_stamp_wscc_emt_grid()
    if full_nonlinear:
        devices = {generator.name: generator for generator in grid.generators}
        for parameters in (STAMP_GFOR, STAMP_GFOL):
            device = devices[f"STAMP {parameters.mode}{parameters.number}"]
            model = build_stamp_nonlinear_converter_emt(
                grid.var_factory, parameters,
                f"STAMP_{parameters.mode}{parameters.number}_EMT",
            )
            set_emt_model(device, model.block, grid.var_factory)
    return grid


def run_case(full_nonlinear: bool, time_step: float, settling_time: float):
    from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
    from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import SmallSignalStabilityEmtDriver
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import SmallSignalStabilityEmtOptions
    from VeraGridEngine.enumerations import DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes

    grid = build_grid(full_nonlinear)
    pf_options = PowerFlowOptions(retry_with_other_methods=True)
    pf = PowerFlowDriver(grid, pf_options)
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid, pf_options)
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError("IEEE-9 power flow did not converge")

    emt_options = EmtOptions(
        time_step=time_step,
        simulation_time=1.0 / grid.fBase,
        tolerance=1e-8,
        solver_type=EmtSolverTypes.StructuralCompiled,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        verbose=0,
    )
    ssa_options = SmallSignalStabilityEmtOptions(
        k=30,
        target_period=1.0 / grid.fBase,
        ss_assessment_time=settling_time,
        verbose=1,
    )
    driver = SmallSignalStabilityEmtDriver(
        grid=grid, emt_options=emt_options, sss_options=ssa_options,
        pf_results=pf.results, pf_results_3ph=pf3.results,
    )
    ssa_options.k = driver.problem.get_states_number()  # Dense full spectrum.
    driver.run()
    results = driver.results
    trajectory = driver.last_period_trajectory
    if results is None or trajectory is None:
        raise RuntimeError("SSA did not retain results and the final-period orbit")
    multipliers = np.asarray(results.multipliers)
    exponents = np.asarray(results.eigenvalues)
    orbit_defect = trajectory[-1] - trajectory[0]
    variables = list(driver.problem.get_state_vars()) + list(driver.problem.get_algebraic_vars())
    for index, variable in enumerate(variables):
        if variable.name.endswith(".theta_grid"):
            orbit_defect[index] = np.angle(np.exp(1j * orbit_defect[index]))
    orbit_scale = max(1.0, float(np.max(np.abs(trajectory[[0, -1]]))))
    label = "full_nonlinear" if full_nonlinear else "linear_deviation"
    output = HERE / f"ieee9_stamp_{label}_comparison_modes.npz"
    np.savez(output, multipliers=multipliers, exponents=exponents,
             state_names=np.asarray(results.stat_vars_array, dtype=str),
             orbit_defect=orbit_defect, time_step=time_step,
             settling_time=settling_time)
    print(f"{label}: modes={multipliers.size}, max|mu|={np.max(np.abs(multipliers)):.9e}, "
          f"unstable={np.count_nonzero(np.abs(multipliers) > 1.0+1e-6)}, "
          f"relative orbit defect={np.max(np.abs(orbit_defect))/orbit_scale:.3e}")
    print(f"data={output}")
    return multipliers, exponents


def plot_comparison(linear, nonlinear) -> Path:
    linear_mu, linear_lam = linear
    nonlinear_mu, nonlinear_lam = nonlinear
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    angle = np.linspace(0.0, 2.0 * np.pi, 500)
    axes[0].plot(np.cos(angle), np.sin(angle), "k--", linewidth=1, label="Unit circle")
    axes[0].scatter(linear_mu.real, linear_mu.imag, s=45, facecolors="none",
                    edgecolors="tab:blue", label="Linear deviation controls")
    axes[0].scatter(nonlinear_mu.real, nonlinear_mu.imag, s=28, marker="x",
                    color="tab:orange", label="Full nonlinear controls")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set(xlabel="Re(mu)", ylabel="Im(mu)", title="Floquet multipliers")
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].scatter(linear_lam.real, linear_lam.imag / (2.0 * np.pi), s=45,
                    facecolors="none", edgecolors="tab:blue", label="Linear deviation controls")
    axes[1].scatter(nonlinear_lam.real, nonlinear_lam.imag / (2.0 * np.pi), s=28,
                    marker="x", color="tab:orange", label="Full nonlinear controls")
    axes[1].set(xlabel="Re(lambda) [1/s]", ylabel="frequency [Hz]",
                title="Floquet exponents")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    output = HERE / "ieee9_stamp_full_nonlinear_vs_linear_deviation.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-step", type=float, default=20e-6)
    parser.add_argument("--settling-time", type=float, default=0.06)
    parser.add_argument("--case", choices=("both", "full-nonlinear", "linear-deviation"),
                        default="both")
    args = parser.parse_args()
    linear = nonlinear = None
    if args.case in ("both", "linear-deviation"):
        linear = run_case(False, args.time_step, args.settling_time)
    if args.case in ("both", "full-nonlinear"):
        nonlinear = run_case(True, args.time_step, args.settling_time)
    if linear is not None and nonlinear is not None:
        nearest = np.min(np.abs(nonlinear[0][:, None] - linear[0][None, :]), axis=1)
        print(f"nonlinear-to-linear nearest |delta mu|: "
              f"median={np.median(nearest):.3e}, max={np.max(nearest):.3e}")
        print(f"plot={plot_comparison(linear, nonlinear)}")


if __name__ == "__main__":
    main()
