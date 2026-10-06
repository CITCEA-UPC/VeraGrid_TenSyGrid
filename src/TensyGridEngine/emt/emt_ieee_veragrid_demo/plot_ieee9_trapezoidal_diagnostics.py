"""Plot the trapezoidal IEEE9 Floquet convergence diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "ieee9_trapezoidal_600_modes.npz"
OUTPUT = HERE / "ieee9_trapezoidal_600_diagnostics.png"


def main() -> None:
    if DATA.exists():
        with np.load(DATA) as data:
            multipliers = np.asarray(data["multipliers"])
            exponents = np.asarray(data["eigenvalues"])
    else:
        # The historical convergence summary below is still useful when the
        # optional saved Ritz spectrum is not present in a checkout.
        multipliers = np.asarray([], dtype=complex)
        exponents = np.asarray([], dtype=complex)

    # Identical solver settings were used in the three resolution checks apart
    # from the expanded 600-step Krylov space. Values are diagnostics, not a
    # converged stability boundary.
    steps = np.asarray([100, 300, 600])
    dominant = np.asarray([1.3928517337, 1.1427288239, 1.0640425627])
    best_residual = np.asarray([3.591e-3, 3.223e-2, 4.299e-1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    angle = np.linspace(0.0, 2.0 * np.pi, 600)
    axes[0].plot(np.cos(angle), np.sin(angle), "k--", lw=1, label="unit circle")
    axes[0].scatter(multipliers.real, multipliers.imag, color="tab:red", zorder=3,
                    label="600-step Ritz values")
    axes[0].set(xlabel="Re(mu)", ylabel="Im(mu)", title="Floquet multipliers")
    axes[0].set_aspect("equal", adjustable="box")
    if multipliers.size:
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "optional 600-step spectrum not available",
                     transform=axes[0].transAxes, ha="center", va="center")

    axes[1].axvline(0.0, color="black", linestyle="--", lw=1)
    axes[1].scatter(exponents.real, exponents.imag, color="tab:purple")
    axes[1].set(xlabel="Re(lambda) [1/s]", ylabel="Im(lambda) [rad/s]",
                title="Floquet exponents (unconverged)")
    if not exponents.size:
        axes[1].text(0.5, 0.5, "optional 600-step spectrum not available",
                     transform=axes[1].transAxes, ha="center", va="center")

    axes[2].axhline(1.0, color="black", linestyle="--", lw=1, label="unit circle")
    axes[2].plot(steps, dominant, "o-", color="tab:blue", label="largest returned |mu|")
    for x, y, residual in zip(steps, dominant, best_residual):
        axes[2].annotate(f"residual {residual:.2g}", (x, y), xytext=(4, 7),
                         textcoords="offset points", fontsize=8)
    axes[2].set(xlabel="Integration steps per 20 ms period", ylabel="Largest returned |mu|",
                title="Resolution/Krylov sensitivity")
    axes[2].legend()

    for axis in axes:
        axis.grid(alpha=0.3)
    fig.suptitle("IEEE9 trapezoidal Floquet diagnostic: Ritz values are not converged")
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
