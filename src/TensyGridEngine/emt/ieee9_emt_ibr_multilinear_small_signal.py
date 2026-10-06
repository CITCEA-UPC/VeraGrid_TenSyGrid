"""Floquet SSA for IEEE9 with one GFL and one GFM, using exact EMT multilinear matrices."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ieee9_emt_ibr_simulation as case
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
    SmallSignalStabilityEmtDriver,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import (
    SmallSignalStabilityEmtOptions,
)
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.enumerations import SmallSignalEmtBuildTypes


FREQUENCY_HZ = 60.0
PERIOD = 1.0 / FREQUENCY_HZ
STEPS_PER_PERIOD = 600
N_MODES = 12


def main() -> None:
    case.TIME_STEP = PERIOD / STEPS_PER_PERIOD
    case.SIMULATION_TIME = 2.0 * case.TIME_STEP
    problem, *_ = case.run_case(1, 1, multilinear_inverters=True)
    phi, structure = problem.linearize_matrices()
    print(
        f"problem={type(problem).__name__} Phi={phi.shape}/{phi.nnz} "
        f"S={structure.shape}/{structure.nnz}"
    )

    emt_options = problem.options
    emt_options.time_step = case.TIME_STEP
    emt_options.simulation_time = PERIOD
    sss_options = SmallSignalStabilityEmtOptions(
        k=N_MODES,
        target_period=PERIOD,
        max_krylov_dim=30,
        ss_assessment_time=PERIOD,
        verbose=1,
        max_restarts=0,
        build_type=SmallSignalEmtBuildTypes.Arnoldi,
        prefer_ak_operator=True,
    )
    power_flow = PowerFlowDriver(grid=problem.grid, options=case.build_power_flow_options())
    power_flow.run()
    driver = SmallSignalStabilityEmtDriver(
        grid=problem.grid,
        emt_options=emt_options,
        sss_options=sss_options,
        pf_results=power_flow.results,
    )
    driver.problem = problem
    driver.run()
    if driver.results is None:
        raise RuntimeError("IBR Floquet analysis returned no results")
    results = driver.results

    directory = Path(__file__).with_name("results")
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "ieee9_emt_ibr_multilinear_modes.csv"
    multiplier_path = directory / "ieee9_emt_ibr_multilinear_multipliers.png"
    exponent_path = directory / "ieee9_emt_ibr_multilinear_eigenvalues.png"
    order = np.argsort(-np.abs(results.multipliers))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "mu_real", "mu_imag", "mu_abs", "lambda_real", "lambda_imag", "frequency_hz"))
        for rank, index in enumerate(order):
            mu, lam = results.multipliers[index], results.eigenvalues[index]
            writer.writerow((rank, mu.real, mu.imag, abs(mu), lam.real, lam.imag, abs(lam.imag) / (2 * np.pi)))

    angle = np.linspace(0.0, 2.0 * np.pi, 500)
    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    ax.plot(np.cos(angle), np.sin(angle), "k--", linewidth=1, label="unit circle")
    stable = np.abs(results.multipliers) <= 1.0
    ax.scatter(results.multipliers.real[stable], results.multipliers.imag[stable], label="stable")
    ax.scatter(results.multipliers.real[~stable], results.multipliers.imag[~stable], marker="x", label="unstable")
    ax.set(xlabel="Re(mu)", ylabel="Im(mu)", title="IEEE9 1-GFL/1-GFM multilinear Floquet multipliers")
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.3); ax.legend()
    fig.savefig(multiplier_path, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="stability boundary")
    stable = results.eigenvalues.real <= 0.0
    ax.scatter(results.eigenvalues.real[stable], results.eigenvalues.imag[stable], label="stable")
    ax.scatter(results.eigenvalues.real[~stable], results.eigenvalues.imag[~stable], marker="x", label="unstable")
    ax.set(xlabel="Re(lambda) [1/s]", ylabel="Im(lambda) [rad/s]", title="IEEE9 1-GFL/1-GFM multilinear Floquet exponents")
    ax.grid(alpha=.3); ax.legend()
    fig.savefig(exponent_path, dpi=180); plt.close(fig)

    print(results.report_stability())
    for index in order:
        print(f"mode={index:02d} |mu|={abs(results.multipliers[index]):.9e} lambda={results.eigenvalues[index]:+.9e}")
    print(f"csv={csv_path}\nmultipliers={multiplier_path}\neigenvalues={exponent_path}")


if __name__ == "__main__":
    main()
