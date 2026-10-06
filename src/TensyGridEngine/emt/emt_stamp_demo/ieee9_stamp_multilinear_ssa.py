#!/usr/bin/env python3
"""Run STAMP GFOR/GFOL Floquet SSA through EmtProblemMultilinear."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
from TensyGridEngine.emt.emt_stamp_demo.stamp_common import add_stamp_public_to_path

add_stamp_public_to_path()

from TensyGridEngine.emt.emt_stamp_demo.stamp_multilinear_converters import build_stamp_converter_emt_multilinear
from TensyGridEngine.emt.emt_stamp_demo.stamp_multilinear_generator import build_stamp_generator_emt_multilinear
from veragrid_stamp.emt_case import build_stamp_wscc_emt_grid
from veragrid_stamp.parameters import STAMP_GFOL, STAMP_GFOR, STAMP_SG


def plot_saved_comparison() -> Path | None:
    """Overlay the two saved spectra once both comparison runs exist."""
    ml_path = HERE / "ieee9_stamp_multilinear_modes.npz"
    ref_path = HERE / "ieee9_stamp_nonlinear_reference_modes.npz"
    if not ml_path.exists() or not ref_path.exists():
        return None

    with np.load(ml_path) as data:
        ml_mu = data["multipliers"]
        ml_lam = data["exponents"]
    with np.load(ref_path) as data:
        ref_mu = data["multipliers"]
        ref_lam = data["exponents"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    angle = np.linspace(0.0, 2.0*np.pi, 500)
    axes[0].plot(np.cos(angle), np.sin(angle), "k--", linewidth=1,
                 label="Unit circle")
    axes[0].scatter(ref_mu.real, ref_mu.imag, s=52, facecolors="none",
                    edgecolors="tab:blue", linewidths=1.5,
                    label="Original controls")
    axes[0].scatter(ml_mu.real, ml_mu.imag, s=34, marker="x",
                    color="tab:orange", linewidths=1.5,
                    label="Multilinear controls")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set(xlabel="Re(mu)", ylabel="Im(mu)",
                title="Floquet multipliers (superposed)")

    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].scatter(ref_lam.real, ref_lam.imag/(2*np.pi), s=52,
                    facecolors="none", edgecolors="tab:blue", linewidths=1.5,
                    label="Original controls")
    axes[1].scatter(ml_lam.real, ml_lam.imag/(2*np.pi), s=34, marker="x",
                    color="tab:orange", linewidths=1.5,
                    label="Multilinear controls")
    axes[1].set(xlabel="Re(lambda) [1/s]", ylabel="frequency [Hz]",
                title="Floquet exponents (superposed)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    output = HERE / "ieee9_stamp_original_vs_multilinear_ssa.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)

    nearest_mu = np.min(np.abs(ml_mu[:, None] - ref_mu[None, :]), axis=1)
    nearest_lam = np.min(np.abs(ml_lam[:, None] - ref_lam[None, :]), axis=1)
    print(f"comparison median nearest |delta mu|={np.median(nearest_mu):.3e}, "
          f"median nearest |delta lambda|={np.median(nearest_lam):.3e}")
    return output


def build_grid(multilinear_converters: bool = True):
    from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model

    grid = build_stamp_wscc_emt_grid()
    devices = {generator.name: generator for generator in grid.generators}
    sg = devices["STAMP SG1"]
    sg_model = build_stamp_generator_emt_multilinear(
        grid.var_factory, STAMP_SG, name="STAMP_SG1_EMT_ML")
    set_emt_model(sg, sg_model.block, grid.var_factory)
    reference_omega = next(variable for variable in sg.emt_model.state_vars
                           if variable.name.endswith(".w_pu"))
    for parameters in (STAMP_GFOR, STAMP_GFOL):
        device = devices[f"STAMP {parameters.mode}{parameters.number}"]
        if multilinear_converters:
            model = build_stamp_converter_emt_multilinear(
                grid.var_factory, parameters,
                name=f"STAMP_{parameters.mode}{parameters.number}_EMT_ML",
                reference_omega=reference_omega)
        else:
            from veragrid_stamp.emt_converters import build_stamp_converter_emt
            model = build_stamp_converter_emt(
                grid.var_factory, parameters,
                name=f"STAMP_{parameters.mode}{parameters.number}_EMT_REF",
                reference_omega=reference_omega)
        set_emt_model(device, model.block, grid.var_factory)
    return grid


def main() -> None:
    import argparse
    from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
    from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import SmallSignalStabilityEmtDriver
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import SmallSignalStabilityEmtOptions
    from VeraGridEngine.enumerations import (DynamicIntegrationMethod, EmtInitializationMethod,
                                             EmtProblemTypes, EmtSolverTypes)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonlinear-converters", action="store_true")
    parser.add_argument("--settling-time", type=float, default=0.06,
                        help="Time simulated before capturing the final period [s]")
    args = parser.parse_args()
    grid = build_grid(multilinear_converters=not args.nonlinear_converters)
    pf_options = PowerFlowOptions(retry_with_other_methods=True)
    pf = PowerFlowDriver(grid, pf_options); pf.run()
    pf3 = PowerFlowDriver3Ph(grid, pf_options); pf3.run()
    options = EmtOptions(
        time_step=20e-6, simulation_time=0.02, tolerance=1e-8,
        solver_type=EmtSolverTypes.StructuralCompiled,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        problem_type=EmtProblemTypes.Multilinear,
        verbose=0)
    ssa_options = SmallSignalStabilityEmtOptions(
        k=30, target_period=1.0/grid.fBase,
        ss_assessment_time=args.settling_time, verbose=1)
    driver = SmallSignalStabilityEmtDriver(
        grid=grid, emt_options=options, sss_options=ssa_options,
        pf_results=pf.results, pf_results_3ph=pf3.results)
    problem = driver.problem
    # Request every physical-state multiplier.  This selects the driver's
    # dense monodromy path and avoids judging the model from only the 30
    # largest-magnitude Arnoldi modes.
    ssa_options.k = problem.get_states_number()
    if not args.nonlinear_converters:
        phi, structure = problem.linearize_matrices()
        print(f"exact multilinear matrices: Phi={phi.shape}, S={structure.shape}")
    driver.run()
    results = driver.results
    multipliers = np.asarray(results.multipliers)
    exponents = np.asarray(results.eigenvalues)
    print(f"problem={type(problem).__name__}, max|mu|={np.max(np.abs(multipliers)):.9e}, "
          f"unstable={np.count_nonzero(np.abs(multipliers) > 1.0+1e-6)}")
    trajectory = driver.last_period_trajectory
    derivative_trajectory = driver.last_period_derivative_trajectory
    monodromy = driver.last_monodromy_matrix
    if trajectory is None or derivative_trajectory is None or monodromy is None:
        raise RuntimeError("Full-spectrum diagnostics were not retained by the driver")
    orbit_defect = trajectory[-1] - trajectory[0]
    trajectory_vars = list(problem.get_state_vars()) + list(problem.get_algebraic_vars())
    periodic_angle_indices = [
        index for index, variable in enumerate(trajectory_vars)
        if variable.name.endswith(".theta_grid")
    ]
    wrapped_orbit_defect = orbit_defect.copy()
    wrapped_orbit_defect[periodic_angle_indices] = np.angle(
        np.exp(1j * wrapped_orbit_defect[periodic_angle_indices]))
    orbit_scale = max(1.0, float(np.max(np.abs(trajectory[[0, -1]]))))
    derivative_defect = derivative_trajectory[-1] - derivative_trajectory[0]
    derivative_scale = max(1.0, float(np.max(np.abs(derivative_trajectory[[0, -1]]))))
    eigenpair_defects = monodromy @ results.right_vecs - results.right_vecs * multipliers
    eigenpair_denominator = np.maximum(
        np.linalg.norm(monodromy, ord=2) * np.linalg.norm(results.right_vecs, axis=0),
        np.finfo(float).eps)
    relative_residuals = np.linalg.norm(eigenpair_defects, axis=0) / eigenpair_denominator
    print(f"full spectrum={multipliers.size}/{monodromy.shape[0]} physical modes")
    print(f"periodic-orbit defect (angles modulo 2pi): "
          f"|dy|inf={np.max(np.abs(wrapped_orbit_defect)):.3e}, "
          f"relative={np.max(np.abs(wrapped_orbit_defect))/orbit_scale:.3e}")
    print(f"periodic-derivative defect: |dydot|inf={np.max(np.abs(derivative_defect)):.3e}, "
          f"relative={np.max(np.abs(derivative_defect))/derivative_scale:.3e}")
    print(f"eigenpair relative residuals: max={np.max(relative_residuals):.3e}, "
          f"median={np.median(relative_residuals):.3e}")
    for index in np.argsort(np.abs(multipliers))[::-1][:12]:
        print(f"  mu={multipliers[index]:+.9e}, lambda={exponents[index]:+.9e}")
    dominant = int(np.argmax(np.abs(multipliers)))
    print("dominant participants:")
    for state_index in np.argsort(results.participation_factors[:, dominant])[::-1][:12]:
        print(f"  {results.participation_factors[state_index, dominant]:.4e} "
              f"{results.stat_vars_array[state_index]}")
    result_suffix = "nonlinear_reference" if args.nonlinear_converters else "multilinear"
    np.savez(HERE / f"ieee9_stamp_{result_suffix}_modes.npz",
             multipliers=multipliers, exponents=exponents,
             state_names=np.asarray(results.stat_vars_array, dtype=str),
             participation=np.asarray(results.participation_factors),
             monodromy=monodromy, orbit_defect=orbit_defect,
             wrapped_orbit_defect=wrapped_orbit_defect,
             derivative_orbit_defect=derivative_defect,
             eigenpair_relative_residuals=relative_residuals)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    angle = np.linspace(0.0, 2.0*np.pi, 500)
    axes[0].plot(np.cos(angle), np.sin(angle), "k--")
    axes[0].scatter(multipliers.real, multipliers.imag, s=24)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set(xlabel="Re(mu)", ylabel="Im(mu)", title="Floquet multipliers")
    axes[1].axvline(0.0, color="black", linestyle="--")
    axes[1].scatter(exponents.real, exponents.imag/(2*np.pi), s=24)
    axes[1].set(xlabel="Re(lambda) [1/s]", ylabel="frequency [Hz]",
                title="Floquet exponents")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    output = HERE / f"ieee9_stamp_{result_suffix}_ssa.png"
    figure.savefig(output, dpi=180); plt.close(figure)
    print(f"plot={output}")
    comparison_output = plot_saved_comparison()
    if comparison_output is not None:
        print(f"comparison_plot={comparison_output}")


if __name__ == "__main__":
    main()
