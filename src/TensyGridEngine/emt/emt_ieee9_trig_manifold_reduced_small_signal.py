"""IEEE9 Floquet experiment with per-step trig-manifold DAE reduction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla

from TensyGridEngine.emt.emt_ieee9_projected_trig_small_signal import _build_driver
from TensyGridEngine.emt.emt_ieee9_small_signal import PERIOD, TIME_STEP
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.manifold_reduced_emt_floquet_operator import (
    TrigManifoldReducedEmtFloquetOperator,
)


def _find_trig_triplets(state_vars) -> list[tuple[int, int, int]]:
    names = np.asarray([str(variable) for variable in state_vars])
    theta = np.flatnonzero(names == "theta_abs_")
    cosine = np.flatnonzero(names == "u_cos_sauer_pai")
    sine = np.flatnonzero(names == "u_sin_sauer_pai")
    if not (len(theta) == len(cosine) == len(sine)) or not len(theta):
        raise RuntimeError(f"Cannot pair trig states: theta={len(theta)}, cos={len(cosine)}, sin={len(sine)}")
    return list(zip(theta.tolist(), cosine.tolist(), sine.tolist()))


def run(n_modes: int):
    # Keep the original controls here so this experiment isolates only the
    # Sauer--Pai trigonometric state lift.
    driver = _build_driver(multilinear_controls=False)
    trajectory, times, jacobian, parameters, n_event = driver._capture_limit_cycle_and_evaluator(
        TIME_STEP, verbose=1
    )
    problem = driver.problem
    triplets = _find_trig_triplets(problem.get_state_vars())
    operator = TrigManifoldReducedEmtFloquetOperator(
        problem=problem,
        trajectory=trajectory,
        h=TIME_STEP,
        n_states=problem.get_states_number(),
        trig_triplets=triplets,
        method=driver.emt_options.integration_method,
        jac_evaluator=jacobian,
        static_params=parameters,
        n_event_params=n_event,
        t_trajectory=times,
    )
    search_k = min(max(2 * n_modes, n_modes + 2), operator.shape[0] - 2)
    multipliers, _ = spla.eigs(operator, k=search_k, which="LM", tol=1e-8)
    keep = np.argsort(-np.abs(multipliers))[:n_modes]
    multipliers = multipliers[keep]
    exponents = np.log(multipliers.astype(complex)) / PERIOD
    return multipliers, exponents, len(triplets), problem.get_states_number(), operator.shape[0]


def save(multipliers: np.ndarray, exponents: np.ndarray) -> tuple[Path, Path]:
    directory = Path(__file__).resolve().parent
    csv_path = directory / "ieee9_emt_trig_manifold_reduced_modes.csv"
    plot_path = directory / "ieee9_emt_trig_manifold_reduced_vs_original_machine.png"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "mu_real", "mu_imag", "mu_abs", "lambda_real", "lambda_imag", "frequency_hz"))
        for rank, index in enumerate(np.argsort(-np.abs(multipliers))):
            value = exponents[index]
            writer.writerow((rank, multipliers[index].real, multipliers[index].imag, abs(multipliers[index]),
                             value.real, value.imag, abs(value.imag) / (2.0 * np.pi)))

    baseline_data = np.genfromtxt(directory / "ieee9_emt_modes_original_machine.csv", delimiter=",", names=True)
    baseline = baseline_data["lambda_real"] + 1j * baseline_data["lambda_imag"]
    figure, axis = plt.subplots(figsize=(9.0, 6.5), constrained_layout=True)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.scatter(baseline.real, baseline.imag, marker="o", facecolors="none", edgecolors="tab:blue",
                 linewidths=1.6, s=78, label="original non-lifted machine")
    axis.scatter(exponents.real, exponents.imag, marker="x", color="tab:orange", linewidths=1.7,
                 s=65, label="per-step trig reduction")
    axis.set_xlabel("Re(lambda) [1/s]")
    axis.set_ylabel("Im(lambda) [rad/s]")
    axis.set_title("IEEE9 EMT: per-step trig reduction vs original machine")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return csv_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=int, default=24)
    args = parser.parse_args()
    multipliers, exponents, count, full_size, reduced_size = run(args.modes)
    csv_path, plot_path = save(multipliers, exponents)
    print(f"trig_lifts={count} full_states={full_size} reduced_states={reduced_size}")
    print(f"spectral_radius={np.max(np.abs(multipliers)):.10f}")
    print(f"csv={csv_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
