"""Compare nonlinear and multilinear IEEE9 EMT Floquet modes."""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
    SmallSignalStabilityEmtDriver,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import (
    SmallSignalStabilityEmtOptions,
)
from VeraGridEngine.enumerations import DynamicIntegrationMethod, SmallSignalEmtBuildTypes

from ieee9_common import GOVERNOR_K, build_ieee9_case


N_MODES = 12
STEPS_PER_PERIOD = int(os.environ.get("IEEE9_SSA_STEPS_PER_PERIOD", "600"))
ASSESSMENT_TIME = float(os.environ.get("IEEE9_SSA_ASSESSMENT_TIME", "0.2"))
MAX_KRYLOV_DIM = int(os.environ.get("IEEE9_SSA_MAX_KRYLOV_DIM", "120"))
MAX_RESTARTS = int(os.environ.get("IEEE9_SSA_MAX_RESTARTS", "3"))
OUTPUT = Path(__file__).with_name("ieee9_small_signal_compare.png")


def modes(multilinear: bool):
    """Build one formulation and calculate its dominant Floquet modes."""
    period = 1.0 / 50.0
    case = build_ieee9_case(
        multilinear,
        period / STEPS_PER_PERIOD,
        ASSESSMENT_TIME,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
    )
    if multilinear:
        phi, structure = case.problem.linearize_matrices()
        print(f"multilinear matrices: Phi={phi.shape}, S={structure.shape}")

    options = SmallSignalStabilityEmtOptions(
        k=N_MODES,
        target_period=period,
        ss_assessment_time=ASSESSMENT_TIME,
        max_krylov_dim=MAX_KRYLOV_DIM,
        max_restarts=MAX_RESTARTS,
        build_type=SmallSignalEmtBuildTypes.HybridArnoldi,
        prefer_ak_operator=False,
        verbose=1,
    )
    driver = SmallSignalStabilityEmtDriver(
        grid=case.grid,
        emt_options=case.problem.options,
        sss_options=options,
        pf_results=case.power_flow,
    )
    driver.problem = case.problem
    driver.run()
    if driver.results is None:
        raise RuntimeError("EMT small-signal analysis returned no results")
    return case, driver.results


def main() -> None:
    reference_case, reference = modes(multilinear=False)
    multilinear_case, multilinear = modes(multilinear=True)
    print(f"reference problem:   {type(reference_case.problem).__name__}")
    print(f"multilinear problem: {type(multilinear_case.problem).__name__}")
    for label, result in (("reference", reference), ("multilinear", multilinear)):
        dominant = int(np.argmax(np.abs(result.multipliers)))
        unstable = int(np.count_nonzero(np.abs(result.multipliers) > 1.0 + 1.0e-6))
        print(
            f"{label}: modes={len(result.multipliers)}, unstable={unstable}, "
            f"dominant |mu|={abs(result.multipliers[dominant]):.6e}"
        )
        participants = np.argsort(result.participation_factors[:, dominant])[-5:][::-1]
        print(f"{label} dominant-mode participants:")
        for state_index in participants:
            print(
                f"  {result.stat_vars_array[state_index]}: "
                f"{result.participation_factors[state_index, dominant]:.3e}"
            )
        for mode_index in np.flatnonzero(np.abs(result.multipliers) > 1.0 + 1.0e-6):
            mode_participants = np.argsort(
                result.participation_factors[:, mode_index]
            )[-3:][::-1]
            names = ", ".join(result.stat_vars_array[mode_participants])
            print(
                f"  unstable mode {mode_index}: mu={result.multipliers[mode_index]:+.6e}, "
                f"lambda={result.eigenvalues[mode_index]:+.6e}, top=[{names}]"
            )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    angle = np.linspace(0.0, 2.0 * np.pi, 500)
    axes[0].plot(np.cos(angle), np.sin(angle), "k--", label="unit circle")
    axes[0].scatter(reference.multipliers.real, reference.multipliers.imag,
                    facecolors="none", edgecolors="tab:blue", label="nonlinear reference")
    axes[0].scatter(multilinear.multipliers.real, multilinear.multipliers.imag,
                    marker="x", color="tab:orange", label="multilinear")
    axes[0].set(xlabel="Re(mu)", ylabel="Im(mu)", title="Floquet multipliers")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].axvline(0.0, color="black", linestyle="--", label="stability boundary")
    axes[1].scatter(reference.eigenvalues.real, reference.eigenvalues.imag,
                    facecolors="none", edgecolors="tab:blue", label="nonlinear reference")
    axes[1].scatter(multilinear.eigenvalues.real, multilinear.eigenvalues.imag,
                    marker="x", color="tab:orange", label="multilinear")
    axes[1].set(xlabel="Re(lambda) [1/s]", ylabel="Im(lambda) [rad/s]",
                title="Floquet exponents")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(f"IEEE9 EMT small-signal comparison (synchronous governor K={GOVERNOR_K:g})")
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(f"plot={OUTPUT}")


if __name__ == "__main__":
    main()
