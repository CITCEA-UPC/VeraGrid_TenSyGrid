"""Sweep governor inverse droop K on the original non-lifted IEEE9 EMT model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla

from TensyGridEngine.emt.emt_ieee9_projected_trig_small_signal import _build_driver
from TensyGridEngine.emt.emt_ieee9_small_signal import PERIOD, TIME_STEP
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_floquet_operator import EmtFloquetOperator


class _GovernorGainFloquetOperator(EmtFloquetOperator):
    """Floquet operator overriding selected runtime governor-gain entries."""

    def __init__(self, *args, gain_indices: list[int], gain: float, **kwargs):
        self._gain_indices = np.asarray(gain_indices, dtype=np.int64)
        self._gain = float(gain)
        super().__init__(*args, **kwargs)

    def _precompute_from_jit(self, jac_evaluator, static_params, n_ev_params, t_trajectory):
        full_params = np.empty(n_ev_params + len(static_params), dtype=float)
        if len(static_params):
            full_params[n_ev_params:] = static_params
        event_params = np.zeros(n_ev_params, dtype=float)
        dx_dummy = np.zeros(self.n_total, dtype=float)
        for step in range(1, len(self.trajectory)):
            x_k = self.trajectory[step]
            x_prev = self.trajectory[step - 1]
            x_prev2 = self.trajectory[step - 2] if step > 1 else x_prev
            x_k = x_k[0] if isinstance(x_k, tuple) else x_k
            x_prev = x_prev[0] if isinstance(x_prev, tuple) else x_prev
            x_prev2 = x_prev2[0] if isinstance(x_prev2, tuple) else x_prev2
            event_params = self.problem.def_event_params_fn(event_params, float(t_trajectory[step]))
            event_params[self._gain_indices] = self._gain
            full_params[:n_ev_params] = event_params
            jacobian = jac_evaluator(states=x_k, params=full_params, history=x_prev,
                                     d_history=dx_dummy, h=self.h, history2=x_prev2)
            self.lu_solvers.append(spla.splu(jacobian))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", type=float, nargs="+", default=[0, 5, 10, 15, 20, 25, 30, 40])
    parser.add_argument("--modes", type=int, default=12)
    args = parser.parse_args()

    driver = _build_driver(multilinear_controls=False, multilinear_machine=False)
    trajectory, times, jacobian, static_parameters, n_event = driver._capture_limit_cycle_and_evaluator(
        TIME_STEP, verbose=1
    )
    problem = driver.problem
    runtime_parameters = problem.get_variable_parameters()
    k_indices = [index for index, parameter in enumerate(runtime_parameters) if parameter.name == "K"]
    if len(k_indices) != 3:
        raise RuntimeError(f"Expected three governor K parameters, found {len(k_indices)}")

    rows: list[tuple[float, complex, complex]] = []
    for gain in args.values:
        operator = _GovernorGainFloquetOperator(
            problem=problem,
            trajectory=trajectory,
            h=TIME_STEP,
            n_states=problem.get_states_number(),
            method=driver.emt_options.integration_method,
            jac_evaluator=jacobian,
            static_params=static_parameters,
            n_event_params=n_event,
            t_trajectory=times,
            gain_indices=k_indices,
            gain=gain,
        )
        search_k = min(max(2 * args.modes, args.modes + 2), operator.shape[0] - 2)
        multipliers, _ = spla.eigs(operator, k=search_k, which="LM", tol=1e-8)
        dominant_index = int(np.argmax(np.abs(multipliers)))
        dominant_mu = multipliers[dominant_index]
        dominant_lambda = np.log(complex(dominant_mu)) / PERIOD
        rows.append((gain, dominant_mu, dominant_lambda))
        print(f"K={gain:g} rho={abs(dominant_mu):.10f} lambda={dominant_lambda:+.9e}")

    directory = Path(__file__).resolve().parent
    csv_path = directory / "ieee9_emt_original_governor_K_sweep.csv"
    plot_path = directory / "ieee9_emt_original_governor_K_sweep.png"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("K", "mu_real", "mu_imag", "mu_abs", "lambda_real", "lambda_imag", "frequency_hz"))
        for gain, multiplier, exponent in rows:
            writer.writerow((gain, multiplier.real, multiplier.imag, abs(multiplier), exponent.real,
                             exponent.imag, abs(exponent.imag) / (2.0 * np.pi)))

    gains = np.asarray([row[0] for row in rows])
    real_parts = np.asarray([row[2].real for row in rows])
    figure, axis = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="stability boundary")
    axis.plot(gains, real_parts, marker="o", color="tab:blue", label="dominant Re(lambda)")
    axis.set_xlabel("Governor inverse droop K")
    axis.set_ylabel("Dominant Re(lambda) [1/s]")
    axis.set_title("IEEE9 original-machine stability versus governor gain")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    print(f"csv={csv_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
