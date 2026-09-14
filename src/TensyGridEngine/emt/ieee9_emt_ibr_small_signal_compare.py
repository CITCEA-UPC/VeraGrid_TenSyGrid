"""Floquet comparison of nonlinear and multilinear IEEE9 inverter models."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ieee9_emt_ibr_simulation as case
from emt_ieee9 import build_emt_options, build_power_flow_options
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
    SmallSignalStabilityEmtDriver,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import (
    SmallSignalStabilityEmtOptions,
)
from VeraGridEngine.enumerations import SmallSignalEmtBuildTypes


N_MODES = 12
STEPS_PER_PERIOD = 600


def run_floquet(multilinear: bool):
    """Build the initialized IBR case and calculate its dominant modes."""
    problem, _time, _values, _gfl, _gfl_buses, _gfm_buses = case.run_case(
        case.N_GFL, case.N_GFM, multilinear_inverters=multilinear
    )
    grid = problem.grid
    period = 1.0 / grid.fBase
    emt_options = build_emt_options()
    emt_options.time_step = period / STEPS_PER_PERIOD
    emt_options.simulation_time = period
    emt_options.problem_type = problem.options.problem_type

    pf = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    pf.run()
    if not bool(pf.results.converged):
        raise RuntimeError("IEEE9 power flow failed before IBR Floquet analysis")

    ss_options = SmallSignalStabilityEmtOptions(
        k=N_MODES,
        target_period=period,
        max_krylov_dim=max(30, 2 * N_MODES),
        ss_assessment_time=period,
        verbose=1,
        max_restarts=0,
        build_type=SmallSignalEmtBuildTypes.Arnoldi,
    )
    driver = SmallSignalStabilityEmtDriver(
        grid=grid,
        emt_options=emt_options,
        sss_options=ss_options,
        pf_results=pf.results,
    )
    driver.problem = problem
    driver.run()
    if driver.results is None:
        raise RuntimeError("IBR Floquet analysis returned no results")
    return driver.results


def save_results(reference, multilinear) -> tuple[Path, Path]:
    output_dir = Path(__file__).resolve().parent
    stem = f"ieee9_emt_ibr_gfl{case.N_GFL}_gfm{case.N_GFM}"
    plot_path = output_dir / f"{stem}_eigenvalues_compare.png"
    csv_path = output_dir / f"{stem}_eigenvalues_compare.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("formulation", "mode", "mu_real", "mu_imag", "mu_abs",
                         "lambda_real", "lambda_imag", "frequency_hz", "damping_ratio"))
        for label, results in (("nonlinear", reference), ("multilinear", multilinear)):
            for mode in np.argsort(-np.abs(results.multipliers)):
                mu = results.multipliers[mode]
                eigenvalue = results.eigenvalues[mode]
                writer.writerow((label, int(mode), mu.real, mu.imag, abs(mu),
                                 eigenvalue.real, eigenvalue.imag,
                                 results.conjugate_frequencies[mode],
                                 results.damping_ratios[mode]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    angle = np.linspace(0.0, 2.0 * np.pi, 600)
    axes[0].plot(np.cos(angle), np.sin(angle), "k--", linewidth=1.0, label="unit circle")
    axes[0].scatter(reference.multipliers.real, reference.multipliers.imag,
                    s=42, facecolors="none", edgecolors="tab:blue", label="nonlinear")
    axes[0].scatter(multilinear.multipliers.real, multilinear.multipliers.imag,
                    s=30, marker="x", color="tab:orange", label="multilinear")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(r"Re($\mu$)")
    axes[0].set_ylabel(r"Im($\mu$)")
    axes[0].set_title("Floquet multipliers")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.0,
                    label="stability boundary")
    axes[1].scatter(reference.eigenvalues.real, reference.eigenvalues.imag,
                    s=42, facecolors="none", edgecolors="tab:blue", label="nonlinear")
    axes[1].scatter(multilinear.eigenvalues.real, multilinear.eigenvalues.imag,
                    s=30, marker="x", color="tab:orange", label="multilinear")
    axes[1].set_xlabel(r"Re($\lambda$) [1/s]")
    axes[1].set_ylabel(r"Im($\lambda$) [rad/s]")
    axes[1].set_title("Floquet exponents")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle(
        f"IEEE9 EMT inverter modes: GFL={case.N_GFL}, GFM={case.N_GFM}"
    )
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return csv_path, plot_path


def print_summary(label: str, results) -> None:
    dominant = int(np.argmax(np.abs(results.multipliers)))
    mu = results.multipliers[dominant]
    eigenvalue = results.eigenvalues[dominant]
    unstable = int(np.count_nonzero(np.abs(results.multipliers) > 1.0 + 1e-6))
    print(
        f"{label}: modes={len(results.multipliers)}, unstable={unstable}, "
        f"dominant_mu={mu.real:+.9e}{mu.imag:+.9e}j, |mu|={abs(mu):.9e}, "
        f"lambda={eigenvalue.real:+.9e}{eigenvalue.imag:+.9e}j"
    )


def main() -> None:
    reference = run_floquet(multilinear=False)
    multilinear = run_floquet(multilinear=True)
    print_summary("nonlinear", reference)
    print_summary("multilinear", multilinear)
    csv_path, plot_path = save_results(reference, multilinear)
    print(f"csv={csv_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()

