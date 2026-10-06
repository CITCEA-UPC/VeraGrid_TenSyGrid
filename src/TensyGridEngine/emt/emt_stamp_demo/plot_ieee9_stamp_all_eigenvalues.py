#!/usr/bin/env python3
"""Show all saved IEEE-9 STAMP Floquet exponents in the complex plane."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
INPUT = HERE / "ieee9_stamp_multilinear_modes.npz"
OUTPUT = HERE / "ieee9_stamp_multilinear_all_30_complex_plane.png"


def main() -> None:
    with np.load(INPUT) as data:
        eigenvalues = np.asarray(data["exponents"], dtype=complex)

    # Use frequency on the imaginary axis for readable engineering units.
    plane_values = eigenvalues.real + 1j * eigenvalues.imag / (2.0 * np.pi)
    keys = np.column_stack((np.round(plane_values.real, 6),
                            np.round(plane_values.imag, 6)))
    locations, counts = np.unique(keys, axis=0, return_counts=True)

    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)

    # Plot all 30 at their exact coordinates. Repeated values deliberately overlap.
    axis.scatter(plane_values.real, plane_values.imag, s=48,
                 color="tab:blue", edgecolor="white", linewidth=0.6, zorder=2)
    for (real, imag), count in zip(locations, counts):
        axis.annotate(f"x{count}", (real, imag), xytext=(6, 6),
                      textcoords="offset points", fontsize=9, fontweight="bold")

    unique_groups = locations.shape[0]
    axis.set(
        xlabel="Re(lambda) [1/s]",
        ylabel="Im(lambda) / 2pi [Hz]",
        title=(f"All {eigenvalues.size} Floquet exponents in the complex plane "
               f"({unique_groups} numerical locations)"),
    )
    axis.grid(True, alpha=0.3)
    axis.text(
        0.01, 0.01,
        "All markers use exact coordinates; xN labels show overlap multiplicity.",
        transform=axis.transAxes, fontsize=9, color="0.3",
    )

    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(f"modes={eigenvalues.size}")
    print(f"plot={OUTPUT}")


if __name__ == "__main__":
    main()
