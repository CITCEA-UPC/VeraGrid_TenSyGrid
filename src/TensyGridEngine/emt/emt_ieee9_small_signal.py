# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Floquet small-signal analysis of the nine-line IEEE9 EMT model."""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt.emt_ieee9 import (
    attach_emt_models,
    build_emt_options,
    build_ieee9_grid,
    build_power_flow_options,
)
from VeraGridEngine.Simulations.EMT.emt_problem_factory import build_emt_problem
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
    SmallSignalStabilityEmtDriver,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import (
    SmallSignalStabilityEmtOptions,
)
from VeraGridEngine.enumerations import SmallSignalEmtBuildTypes


FREQUENCY_HZ = 60.0
PERIOD = 1.0 / FREQUENCY_HZ
STEPS_PER_PERIOD = 600
TIME_STEP = PERIOD / STEPS_PER_PERIOD
N_MODES = 12


def run_small_signal_analysis(
    event_overrides: dict[str, float] | None = None,
    multilinear_controls: bool = True,
    n_modes: int = N_MODES,
):
    """Build the periodic IEEE9 DAE and calculate dominant Floquet modes."""
    grid = build_ieee9_grid()
    attach_emt_models(grid, multilinear_controls=multilinear_controls)
    if event_overrides:
        for generator in grid.generators:
            for parameter_name, value in event_overrides.items():
                generator.emt_model.set_parameter_in_model(parameter_name, float(value))

    power_flow_options = build_power_flow_options()
    balanced_pf = PowerFlowDriver(grid=grid, options=power_flow_options)
    balanced_pf.run()
    three_phase_pf = PowerFlowDriver3Ph(grid=grid, options=power_flow_options)
    three_phase_pf.run()
    if not bool(balanced_pf.results.converged) or not bool(three_phase_pf.results.converged):
        raise RuntimeError("IEEE9 power flow failed before EMT small-signal analysis")

    emt_options = build_emt_options()
    emt_options.time_step = TIME_STEP
    emt_options.simulation_time = PERIOD
    problem = build_emt_problem(
        grid=grid,
        options=emt_options,
        pf_results=balanced_pf.results,
        pf_results_3ph=three_phase_pf.results,
    )

    small_signal_options = SmallSignalStabilityEmtOptions(
        k=n_modes,
        target_period=PERIOD,
        max_krylov_dim=max(30, 2 * n_modes),
        ss_assessment_time=PERIOD,
        verbose=1,
        max_restarts=0,
        build_type=SmallSignalEmtBuildTypes.Arnoldi,
    )
    driver = SmallSignalStabilityEmtDriver(
        grid=grid,
        emt_options=emt_options,
        sss_options=small_signal_options,
        pf_results=balanced_pf.results,
    )
    # The public driver currently accepts one PF result in its constructor.
    # Retain the DAE initialized with both balanced and PF3 operating points.
    driver.problem = problem
    driver.run()
    if driver.results is None:
        raise RuntimeError("EMT small-signal analysis returned no results")
    return driver.results


def save_results(results) -> tuple[Path, Path, Path]:
    """Save the dominant modes and a Floquet multiplier plot."""
    output_directory = Path(__file__).resolve().parent
    csv_path = output_directory / "ieee9_emt_small_signal_modes.csv"
    plot_path = output_directory / "ieee9_emt_floquet_multipliers.png"
    eigenvalue_plot_path = output_directory / "ieee9_emt_eigenvalues.png"

    order = np.argsort(-np.abs(results.multipliers))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "mu_real", "mu_imag", "mu_abs", "eigenvalue_real", "eigenvalue_imag", "frequency_hz", "damping_ratio"))
        for output_index, mode_index in enumerate(order):
            multiplier = results.multipliers[mode_index]
            eigenvalue = results.eigenvalues[mode_index]
            writer.writerow((
                output_index,
                multiplier.real,
                multiplier.imag,
                abs(multiplier),
                eigenvalue.real,
                eigenvalue.imag,
                results.conjugate_frequencies[mode_index],
                results.damping_ratios[mode_index],
            ))

    figure, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    angle = np.linspace(0.0, 2.0 * np.pi, 500)
    axis.plot(np.cos(angle), np.sin(angle), "k--", linewidth=1.0, label="unit circle")
    stable = np.abs(results.multipliers) <= 1.0
    axis.scatter(results.multipliers.real[stable], results.multipliers.imag[stable], s=55, label="stable")
    axis.scatter(results.multipliers.real[~stable], results.multipliers.imag[~stable], s=65, marker="x", label="outside unit circle")
    for index, multiplier in enumerate(results.multipliers):
        axis.annotate(str(index), (multiplier.real, multiplier.imag), xytext=(4, 4), textcoords="offset points")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Re(mu)")
    axis.set_ylabel("Im(mu)")
    axis.set_title("IEEE9 EMT Floquet multipliers")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="stability boundary")
    stable_lambda = results.eigenvalues.real <= 0.0
    axis.scatter(results.eigenvalues.real[stable_lambda], results.eigenvalues.imag[stable_lambda], s=55, label="stable")
    axis.scatter(results.eigenvalues.real[~stable_lambda], results.eigenvalues.imag[~stable_lambda], s=65, marker="x", label="unstable")
    for index, eigenvalue in enumerate(results.eigenvalues):
        axis.annotate(str(index), (eigenvalue.real, eigenvalue.imag), xytext=(4, 4), textcoords="offset points")
    axis.set_xlabel("Re(lambda) [1/s]")
    axis.set_ylabel("Im(lambda) [rad/s]")
    axis.set_title("IEEE9 EMT Floquet exponents")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(eigenvalue_plot_path, dpi=180)
    plt.close(figure)
    return csv_path, plot_path, eigenvalue_plot_path


def report_unstable_mode_participation(results, top_n: int = 15) -> None:
    """Print the states that participate most strongly in each unstable mode."""
    unstable = np.flatnonzero(np.abs(results.multipliers) > 1.0 + 1e-6)
    if unstable.size == 0:
        print("No Floquet multipliers outside the unit circle.")
        return

    state_names = np.asarray(results.stat_vars_array, dtype=str)
    participation = np.asarray(results.participation_factors)
    right_vectors = np.asarray(results.right_vecs)
    for mode_index in unstable:
        multiplier = results.multipliers[mode_index]
        exponent = results.eigenvalues[mode_index]
        pf = participation[:, mode_index]
        shape = np.abs(right_vectors[:, mode_index])
        shape /= max(float(np.max(shape)), np.finfo(float).tiny)
        top = np.argsort(pf)[::-1][:top_n]
        print(
            f"\nUNSTABLE MODE {mode_index}: "
            f"mu={multiplier.real:+.9e}{multiplier.imag:+.9e}j, "
            f"|mu|={abs(multiplier):.9e}, "
            f"lambda={exponent.real:+.9e}{exponent.imag:+.9e}j, "
            f"f={results.conjugate_frequencies[mode_index]:.6f} Hz"
        )
        print("  rank  participation  rel_mode_shape  state")
        for rank, state_index in enumerate(top, start=1):
            print(
                f"  {rank:>4d}  {pf[state_index]:>13.6e}  "
                f"{shape[state_index]:>14.6e}  {state_names[state_index]}"
            )


def main() -> None:
    results = run_small_signal_analysis()
    csv_path, plot_path, eigenvalue_plot_path = save_results(results)
    print(results.report_stability())
    print(f"states={len(results.stat_vars)}, modes={len(results.multipliers)}")
    for index in np.argsort(-np.abs(results.multipliers)):
        multiplier = results.multipliers[index]
        eigenvalue = results.eigenvalues[index]
        print(
            f"mode={index:02d} mu={multiplier.real:+.8e}{multiplier.imag:+.8e}j "
            f"|mu|={abs(multiplier):.8e} "
            f"lambda={eigenvalue.real:+.8e}{eigenvalue.imag:+.8e}j "
            f"f={results.conjugate_frequencies[index]:.6f}Hz "
            f"zeta={results.damping_ratios[index]:+.6e}"
        )
    report_unstable_mode_participation(results)
    print(f"csv={csv_path}")
    print(f"plot={plot_path}")
    print(f"eigenvalue_plot={eigenvalue_plot_path}")


if __name__ == "__main__":
    main()
