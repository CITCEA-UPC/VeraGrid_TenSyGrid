"""Plot the IEEE9 projected trig-manifold modes against both baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _load_exponents(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data["lambda_real"] + 1j * data["lambda_imag"]


def _plot_comparison(
    reference: np.ndarray,
    projected: np.ndarray,
    reference_label: str,
    title: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 6.5), constrained_layout=True)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.scatter(
        reference.real,
        reference.imag,
        marker="o",
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=1.6,
        s=78,
        zorder=2,
    )
    axis.scatter(
        projected.real,
        projected.imag,
        marker="x",
        color="tab:orange",
        linewidths=1.7,
        s=65,
        zorder=3,
    )
    axis.set_xlabel("Re(lambda) [1/s]")
    axis.set_ylabel("Im(lambda) [rad/s]")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend(handles=(
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="tab:blue", markeredgewidth=1.6, markersize=8,
               label=reference_label),
        Line2D([], [], marker="x", linestyle="none", color="tab:orange",
               markeredgewidth=1.7, markersize=8, label="projected multilinear"),
    ))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    directory = Path(__file__).resolve().parent
    projected = _load_exponents(directory / "ieee9_emt_projected_trig_modes.csv")
    unprojected = _load_exponents(directory / "ieee9_emt_modes_ml.csv")
    non_multilinear = _load_exponents(directory / "ieee9_emt_modes_non_ml.csv")

    projected_unprojected = directory / "ieee9_emt_projected_vs_unprojected_eigenvalues.png"
    projected_non_ml = directory / "ieee9_emt_projected_vs_non_multilinear_eigenvalues.png"
    _plot_comparison(
        reference=unprojected,
        projected=projected,
        reference_label="unprojected multilinear",
        title="IEEE9 EMT: projected vs unprojected multilinear modes",
        output=projected_unprojected,
    )
    _plot_comparison(
        reference=non_multilinear,
        projected=projected,
        reference_label="non-multilinear",
        title="IEEE9 EMT: projected multilinear vs non-multilinear modes",
        output=projected_non_ml,
    )
    print(f"projected_vs_unprojected={projected_unprojected}")
    print(f"projected_vs_non_multilinear={projected_non_ml}")


if __name__ == "__main__":
    main()
