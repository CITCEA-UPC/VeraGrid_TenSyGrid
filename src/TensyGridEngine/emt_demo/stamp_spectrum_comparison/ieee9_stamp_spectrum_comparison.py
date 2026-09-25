#!/usr/bin/env python3
"""Compare enlarged IEEE9 STAMP original/multilinear Floquet spectra.

The generator uses its exact multilinear lift in both cases.  ``Original``
means the untouched STAMP GFOR/GFOL converter equations; ``Multilinear`` uses
their exact auxiliary-variable lifts.  Running both cases with the same larger
Krylov request avoids mistaking a truncated Arnoldi spectrum for a missing mode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEMO_ROOT = HERE.parent
STAMP_ROOT = Path("/home/pablo/Desktop/eroots/STAMP_Public")
for path in (DEMO_ROOT, STAMP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ieee9_stamp_multilinear_ssa import build_grid


def run_case(multilinear: bool, krylov_modes: int):
    from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
    from VeraGridEngine.Simulations.EMT.problems.emt_problem_multilinear import EmtProblemMultilinear
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
    from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import SmallSignalStabilityEmtDriver
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import SmallSignalStabilityEmtOptions
    from VeraGridEngine.enumerations import DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes

    grid = build_grid(multilinear_converters=multilinear)
    pf_options = PowerFlowOptions(retry_with_other_methods=True)
    pf = PowerFlowDriver(grid, pf_options)
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid, pf_options)
    pf3.run()

    emt_options = EmtOptions(
        time_step=20e-6,
        simulation_time=0.02,
        tolerance=1e-8,
        solver_type=EmtSolverTypes.StructuralCompiled,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        conventional_three_phase_base=True,
        verbose=0,
    )
    problem = EmtProblemMultilinear(
        grid=grid, options=emt_options,
        pf_results=pf.results, pf_results_3ph=pf3.results,
    )
    if multilinear:
        phi, structure = problem.linearize_matrices()
        print(f"Exact multilinear matrices: Phi={phi.shape}, S={structure.shape}")

    ssa_options = SmallSignalStabilityEmtOptions(
        k=krylov_modes,
        target_period=1.0/grid.fBase,
        ss_assessment_time=0.06,
        verbose=1,
    )
    driver = SmallSignalStabilityEmtDriver(
        grid=grid,
        emt_options=emt_options,
        sss_options=ssa_options,
        pf_results=pf3.results,
    )
    driver.problem = problem
    driver.run()
    return np.asarray(driver.results.multipliers), np.asarray(driver.results.eigenvalues)


def nearest_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.min(np.abs(left[:, None] - right[None, :]), axis=1)


def plot_comparison(ref_mu, ref_lam, ml_mu, ml_lam) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    angle = np.linspace(0.0, 2.0*np.pi, 500)
    axes[0].plot(np.cos(angle), np.sin(angle), "k--", linewidth=1, label="Unit circle")
    axes[0].scatter(ref_mu.real, ref_mu.imag, s=48, facecolors="none",
                    edgecolors="tab:blue", linewidths=1.4, label="Original controls")
    axes[0].scatter(ml_mu.real, ml_mu.imag, s=30, marker="x",
                    color="tab:orange", linewidths=1.4, label="Multilinear controls")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set(xlabel="Re(mu)", ylabel="Im(mu)", title="Floquet multipliers")

    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].scatter(ref_lam.real, ref_lam.imag/(2*np.pi), s=48,
                    facecolors="none", edgecolors="tab:blue", linewidths=1.4,
                    label="Original controls")
    axes[1].scatter(ml_lam.real, ml_lam.imag/(2*np.pi), s=30, marker="x",
                    color="tab:orange", linewidths=1.4, label="Multilinear controls")
    axes[1].set(xlabel="Re(lambda) [1/s]", ylabel="frequency [Hz]",
                title="Floquet exponents")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    output = HERE / "ieee9_stamp_original_vs_multilinear_krylov.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=100,
                        help="Arnoldi/Krylov modes requested for each case (default: 100)")
    args = parser.parse_args()

    print(f"Running original STAMP controls with k={args.k} ...")
    ref_mu, ref_lam = run_case(multilinear=False, krylov_modes=args.k)
    print(f"Running exact multilinear STAMP controls with k={args.k} ...")
    ml_mu, ml_lam = run_case(multilinear=True, krylov_modes=args.k)

    np.savez(HERE / "ieee9_stamp_original_vs_multilinear_krylov.npz",
             original_multipliers=ref_mu, original_exponents=ref_lam,
             multilinear_multipliers=ml_mu, multilinear_exponents=ml_lam)
    mu_distance = nearest_distances(ml_mu, ref_mu)
    lam_distance = nearest_distances(ml_lam, ref_lam)
    print(f"returned modes: original={len(ref_mu)}, multilinear={len(ml_mu)}")
    print(f"median nearest |delta mu|={np.median(mu_distance):.3e}")
    print(f"median nearest |delta lambda|={np.median(lam_distance):.3e}")
    print(f"unstable: original={np.count_nonzero(np.abs(ref_mu) > 1.0+1e-6)}, "
          f"multilinear={np.count_nonzero(np.abs(ml_mu) > 1.0+1e-6)}")
    print(f"plot={plot_comparison(ref_mu, ref_lam, ml_mu, ml_lam)}")


if __name__ == "__main__":
    main()
