"""Compare IEEE9 Floquet modes with original and multilinearized controls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment

from TensyGridEngine.emt.emt_ieee9_small_signal import run_small_signal_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("formulation", choices=("non_ml", "ml", "compare"))
    parser.add_argument("--modes", type=int, default=24)
    args = parser.parse_args()
    output_directory = Path(__file__).resolve().parent
    if args.formulation == "compare":
        non_ml = np.genfromtxt(output_directory / "ieee9_emt_modes_non_ml.csv", delimiter=",", names=True)
        ml = np.genfromtxt(output_directory / "ieee9_emt_modes_ml.csv", delimiter=",", names=True)
        lam_non = non_ml["lambda_real"] + 1j * non_ml["lambda_imag"]
        lam_ml = ml["lambda_real"] + 1j * ml["lambda_imag"]
        rows, cols = linear_sum_assignment(np.abs(lam_non[:, None] - lam_ml[None, :]))
        matched_path = output_directory / "ieee9_emt_mode_matching.csv"
        with matched_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("non_ml_mode", "ml_mode", "non_ml_real", "non_ml_imag", "ml_real", "ml_imag", "absolute_error"))
            for row, col in zip(rows, cols):
                writer.writerow((row, col, lam_non[row].real, lam_non[row].imag,
                                 lam_ml[col].real, lam_ml[col].imag, abs(lam_non[row] - lam_ml[col])))
        plot_path = output_directory / "ieee9_emt_ml_non_ml_eigenvalues.png"
        figure, axis = plt.subplots(figsize=(8.4, 6.2), constrained_layout=True)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        original_color = "tab:blue"
        multilinear_color = "tab:orange"
        axis.scatter(lam_non.real, lam_non.imag, marker="o", facecolors="none",
                     edgecolors=original_color, linewidths=1.5, s=70)
        axis.scatter(lam_ml.real, lam_ml.imag, marker="x", color=multilinear_color,
                     linewidths=1.5, s=55)
        axis.set_xlabel("Re(lambda) [1/s]")
        axis.set_ylabel("Im(lambda) [rad/s]")
        axis.set_title("IEEE9 EMT: original vs multilinear control modes")
        axis.grid(alpha=0.3)
        # Explicit handles keep the hollow original-control marker visible in
        # the legend across Matplotlib versions and output backends.
        axis.legend(handles=(
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                   markeredgecolor=original_color, markeredgewidth=1.5,
                   markersize=8, label="original controls"),
            Line2D([], [], marker="x", linestyle="none", color=multilinear_color,
                   markeredgewidth=1.5, markersize=8,
                   label="multilinear controls"),
        ))
        figure.savefig(plot_path, dpi=180)
        plt.close(figure)
        print(f"matching={matched_path}")
        print(f"plot={plot_path}")
        return

    use_ml = args.formulation == "ml"
    results = run_small_signal_analysis(multilinear_controls=use_ml, n_modes=args.modes)
    output = output_directory / f"ieee9_emt_modes_{args.formulation}.csv"
    order = np.argsort(-np.abs(results.multipliers))
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "mu_real", "mu_imag", "mu_abs", "lambda_real", "lambda_imag", "frequency_hz", "damping_ratio"))
        for rank, index in enumerate(order):
            mu = results.multipliers[index]
            lam = results.eigenvalues[index]
            writer.writerow((rank, mu.real, mu.imag, abs(mu), lam.real, lam.imag,
                             results.conjugate_frequencies[index], results.damping_ratios[index]))
    print(results.report_stability())
    print(f"formulation={args.formulation} states={len(results.stat_vars)} output={output}")


if __name__ == "__main__":
    main()
