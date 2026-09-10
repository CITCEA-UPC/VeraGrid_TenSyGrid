"""IEEE9 Floquet reference using the original non-lifted Sauer--Pai machine."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt.emt_ieee9_projected_trig_small_signal import _build_driver
from TensyGridEngine.emt.emt_ieee9_small_signal import PERIOD, TIME_STEP
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
    SmallSignalStabilityEmtDriver,
)


def main() -> None:
    driver: SmallSignalStabilityEmtDriver = _build_driver(
        multilinear_controls=False,
        multilinear_machine=False,
    )
    driver.sss_options.k = 24
    driver.sss_options.max_krylov_dim = 48
    driver.run()
    if driver.results is None:
        raise RuntimeError("Original-machine Floquet analysis returned no results")
    results = driver.results
    output = Path(__file__).resolve().parent / "ieee9_emt_modes_original_machine.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "mu_real", "mu_imag", "mu_abs", "lambda_real", "lambda_imag", "frequency_hz"))
        for rank, index in enumerate(np.argsort(-np.abs(results.multipliers))):
            mu = results.multipliers[index]
            value = results.eigenvalues[index]
            writer.writerow((rank, mu.real, mu.imag, abs(mu), value.real, value.imag,
                             abs(value.imag) / (2.0 * np.pi)))
    plot_path = Path(__file__).resolve().parent / "ieee9_emt_original_machine_eigenvalues.png"
    figure, axis = plt.subplots(figsize=(9.0, 6.5), constrained_layout=True)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="stability boundary")
    stable = results.eigenvalues.real <= 0.0
    axis.scatter(results.eigenvalues.real[stable], results.eigenvalues.imag[stable],
                 color="tab:blue", s=65, label="stable modes")
    axis.scatter(results.eigenvalues.real[~stable], results.eigenvalues.imag[~stable],
                 color="tab:red", marker="x", linewidths=1.8, s=80, label="unstable modes")
    for index, eigenvalue in enumerate(results.eigenvalues):
        axis.annotate(str(index), (eigenvalue.real, eigenvalue.imag), xytext=(4, 4),
                      textcoords="offset points", fontsize=8)
    axis.set_xlabel("Re(lambda) [1/s]")
    axis.set_ylabel("Im(lambda) [rad/s]")
    axis.set_title("IEEE9 EMT original non-lifted machine Floquet exponents")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(results.report_stability())
    print(f"states={len(results.stat_vars)} period={PERIOD} step={TIME_STEP}")
    print(f"output={output}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
