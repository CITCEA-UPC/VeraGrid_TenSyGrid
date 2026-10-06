"""IEEE9 Floquet analysis projected onto the Sauer--Pai trig manifold.

This is an isolated experiment: it does not modify the generator templates or
the existing IEEE9 scripts.  The lifted ``u_cos/u_sin`` dynamics are retained
for the periodic EMT trajectory, while Floquet perturbations are restricted to
the tangent space of ``u_cos=cos(theta_abs)``, ``u_sin=sin(theta_abs)``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla

from TensyGridEngine.emt.emt_ieee9 import (
    attach_emt_models,
    build_emt_options,
    build_ieee9_grid,
    build_power_flow_options,
)
from TensyGridEngine.emt.emt_ieee9_small_signal import PERIOD, TIME_STEP
from VeraGridEngine.Simulations.EMT.emt_problem_factory import build_emt_problem
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_floquet_operator import EmtFloquetOperator
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
    SmallSignalStabilityEmtDriver,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import (
    SmallSignalStabilityEmtOptions,
)


def _build_driver(
    multilinear_controls: bool = True,
    multilinear_machine: bool = True,
) -> SmallSignalStabilityEmtDriver:
    grid = build_ieee9_grid()
    attach_emt_models(
        grid,
        multilinear_controls=multilinear_controls,
        multilinear_machine=multilinear_machine,
    )
    pf_options = build_power_flow_options()
    pf = PowerFlowDriver(grid=grid, options=pf_options)
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=pf_options)
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError("IEEE9 power flow failed")

    emt_options = build_emt_options()
    emt_options.time_step = TIME_STEP
    emt_options.simulation_time = PERIOD
    problem = build_emt_problem(
        grid=grid,
        options=emt_options,
        pf_results=pf.results,
        pf_results_3ph=pf3.results,
    )
    ss_options = SmallSignalStabilityEmtOptions(
        k=24,
        target_period=PERIOD,
        ss_assessment_time=PERIOD,
        verbose=1,
    )
    driver = SmallSignalStabilityEmtDriver(
        grid=grid,
        emt_options=emt_options,
        sss_options=ss_options,
        pf_results=pf.results,
    )
    driver.problem = problem
    return driver


def _trig_tangent_basis(state_vars, state_point: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    names = np.asarray([str(variable) for variable in state_vars])
    theta = np.flatnonzero(names == "theta_abs_")
    cosine = np.flatnonzero(names == "u_cos_sauer_pai")
    sine = np.flatnonzero(names == "u_sin_sauer_pai")
    if not (len(theta) == len(cosine) == len(sine)) or len(theta) == 0:
        raise RuntimeError(
            f"Could not pair trig states: theta={len(theta)}, cos={len(cosine)}, sin={len(sine)}"
        )

    triples = list(zip(theta.tolist(), cosine.tolist(), sine.tolist()))
    constraints = np.zeros((2 * len(triples), len(state_vars)), dtype=float)
    for machine, (theta_index, cos_index, sin_index) in enumerate(triples):
        u_cos = state_point[cos_index]
        u_sin = state_point[sin_index]
        # du_cos = -sin(theta) dtheta; du_sin = cos(theta) dtheta.
        constraints[2 * machine, theta_index] = u_sin
        constraints[2 * machine, cos_index] = 1.0
        constraints[2 * machine + 1, theta_index] = -u_cos
        constraints[2 * machine + 1, sin_index] = 1.0

    basis = la.null_space(constraints)
    residual = la.norm(constraints @ basis, ord=np.inf)
    if residual > 1e-10:
        raise RuntimeError(f"Trig tangent basis residual is too large: {residual:.3e}")
    return basis, triples


def run_projected_analysis(n_modes: int):
    driver = _build_driver()
    problem = driver.problem
    trajectory, times, jacobian, parameters, n_event = driver._capture_limit_cycle_and_evaluator(
        TIME_STEP, verbose=1
    )
    n_states = problem.get_states_number()
    monodromy = EmtFloquetOperator(
        problem=problem,
        trajectory=trajectory,
        h=TIME_STEP,
        n_states=n_states,
        method=driver.emt_options.integration_method,
        jac_evaluator=jacobian,
        static_params=parameters,
        n_event_params=n_event,
        t_trajectory=times,
    )

    first = trajectory[0][0] if isinstance(trajectory[0], tuple) else trajectory[0]
    basis, triples = _trig_tangent_basis(problem.get_state_vars(), np.asarray(first)[:n_states])
    reduced_size = basis.shape[1]

    projected = spla.LinearOperator(
        shape=(reduced_size, reduced_size),
        dtype=np.float64,
        matvec=lambda vector: basis.T @ monodromy.matvec(basis @ vector),
    )
    search_k = min(max(2 * n_modes, n_modes + 2), reduced_size - 2)
    multipliers, vectors = spla.eigs(projected, k=search_k, which="LM", tol=1e-8)
    keep = np.argsort(-np.abs(multipliers))[:n_modes]
    multipliers = multipliers[keep]
    vectors = basis @ vectors[:, keep]
    exponents = np.log(multipliers.astype(complex)) / PERIOD
    return multipliers, exponents, vectors, triples, n_states, reduced_size


def save_results(multipliers: np.ndarray, exponents: np.ndarray) -> tuple[Path, Path]:
    directory = Path(__file__).resolve().parent
    csv_path = directory / "ieee9_emt_projected_trig_modes.csv"
    plot_path = directory / "ieee9_emt_projected_trig_eigenvalues.png"
    order = np.argsort(-np.abs(multipliers))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "mu_real", "mu_imag", "mu_abs", "lambda_real", "lambda_imag", "frequency_hz"))
        for rank, index in enumerate(order):
            writer.writerow((rank, multipliers[index].real, multipliers[index].imag, abs(multipliers[index]),
                             exponents[index].real, exponents[index].imag,
                             abs(exponents[index].imag) / (2.0 * np.pi)))

    figure, axis = plt.subplots(figsize=(8.4, 6.2), constrained_layout=True)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="stability boundary")
    stable = exponents.real <= 0.0
    axis.scatter(exponents.real[stable], exponents.imag[stable], s=60, label="projected stable modes")
    axis.scatter(exponents.real[~stable], exponents.imag[~stable], marker="x", s=65,
                 label="projected unstable modes")
    axis.set_xlabel("Re(lambda) [1/s]")
    axis.set_ylabel("Im(lambda) [rad/s]")
    axis.set_title("IEEE9 EMT Floquet modes projected onto trig manifold")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return csv_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=int, default=24)
    args = parser.parse_args()
    multipliers, exponents, _, triples, full_size, reduced_size = run_projected_analysis(args.modes)
    csv_path, plot_path = save_results(multipliers, exponents)
    print(f"trig_lifts={len(triples)} full_states={full_size} projected_states={reduced_size}")
    print(f"spectral_radius={np.max(np.abs(multipliers)):.10f}")
    for index in np.argsort(-np.abs(multipliers)):
        print(f"mu={multipliers[index]:+.9e} |mu|={abs(multipliers[index]):.9e} "
              f"lambda={exponents[index]:+.9e}")
    print(f"csv={csv_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
