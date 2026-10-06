"""Screen GFM controller gains against the IEEE9 nonlinear EMT Floquet model."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from TensyGridEngine.emt.emt_ieee_veragrid_demo.ieee9_common import build_ieee9_case
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_floquet_operator import (
    EmtFloquetOperator,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
    SmallSignalStabilityEmtDriver,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import (
    SmallSignalStabilityEmtOptions,
)
from VeraGridEngine.enumerations import DynamicIntegrationMethod


PERIOD = 1.0 / 50.0
STEPS_PER_PERIOD = 600
N_MODES = 12
OUTPUT = Path(__file__).with_name("ieee9_emt_gfm_stability_sweep.csv")

# The PI integrator states enter linearly and their input errors are zero at the
# unsaturated equilibrium, so their gains can also be screened on this trajectory.
CANDIDATES = {
    "baseline": {},
    "Kdp=0": {"Kdp": 0.0},
    "Kdp=.001": {"Kdp": 0.001},
    "Kdp=.002": {"Kdp": 0.002},
    "Kdp=.004": {"Kdp": 0.004},
    "Kdp=.006": {"Kdp": 0.006},
    "Kdq=0": {"Kdq": 0.0},
    "Kdq=.001": {"Kdq": 0.001},
    "Kdq=.010": {"Kdq": 0.010},
    "Kdq=.020": {"Kdq": 0.020},
}


class ParameterOverrideOperator(EmtFloquetOperator):
    """Floquet operator that replaces selected event parameters."""

    def __init__(self, *args, overrides: dict[int, float], **kwargs):
        self.overrides = overrides
        super().__init__(*args, **kwargs)

    def _precompute_from_jit(self, jac_evaluator, static_params, n_event_params, t_trajectory):
        full_params = np.empty(n_event_params + len(static_params), dtype=float)
        full_params[n_event_params:] = static_params
        event_params = np.zeros(n_event_params, dtype=float)
        dx_dummy = np.zeros(self.n_total, dtype=float)
        for step in range(1, len(self.trajectory)):
            current = self.trajectory[step]
            previous = self.trajectory[step - 1]
            previous2 = self.trajectory[step - 2] if step > 1 else previous
            current = current[0] if isinstance(current, tuple) else current
            previous = previous[0] if isinstance(previous, tuple) else previous
            previous2 = previous2[0] if isinstance(previous2, tuple) else previous2
            event_params = self.problem.def_event_params_fn(event_params, float(t_trajectory[step]))
            for index, value in self.overrides.items():
                event_params[index] = value
            full_params[:n_event_params] = event_params
            jacobian = jac_evaluator(
                states=current,
                params=full_params,
                history=previous,
                d_history=dx_dummy,
                h=self.h,
                history2=previous2,
            )
            self.lu_solvers.append(spla.splu(jacobian))


def main() -> None:
    case = build_ieee9_case(
        False,
        PERIOD / STEPS_PER_PERIOD,
        0.2,
        integration_method=DynamicIntegrationMethod.DaeBackEuler,
    )
    driver = SmallSignalStabilityEmtDriver(
        grid=case.grid,
        emt_options=case.problem.options,
        sss_options=SmallSignalStabilityEmtOptions(
            k=N_MODES, target_period=PERIOD, ss_assessment_time=0.2
        ),
        pf_results=case.power_flow,
    )
    driver.problem = case.problem
    trajectory, times, jacobian, static_parameters, n_event = (
        driver._capture_limit_cycle_and_evaluator(PERIOD / STEPS_PER_PERIOD, verbose=1)
    )

    runtime_parameters = case.problem.get_variable_parameters()
    parameter_indices: dict[str, list[int]] = {}
    for wanted in {name for values in CANDIDATES.values() for name in values}:
        matches = [
            i for i, parameter in enumerate(runtime_parameters)
            if parameter.name == wanted or parameter.name.startswith(f"{wanted}_emt_gfm_")
        ]
        expected = 2 if wanted in {"K", "D", "Ks"} else 1
        if len(matches) != expected:
            raise RuntimeError(f"Expected {expected} parameters named {wanted!r}, found {len(matches)}")
        parameter_indices[wanted] = matches

    rows = []
    for label, values in CANDIDATES.items():
        overrides = {
            index: value
            for name, value in values.items()
            for index in parameter_indices[name]
        }
        operator = ParameterOverrideOperator(
            problem=case.problem,
            trajectory=trajectory,
            h=PERIOD / STEPS_PER_PERIOD,
            n_states=case.problem.get_states_number(),
            method=case.problem.options.integration_method,
            jac_evaluator=jacobian,
            static_params=static_parameters,
            n_event_params=n_event,
            t_trajectory=times,
            overrides=overrides,
        )
        search_k = min(2 * N_MODES, operator.shape[0] - 2)
        multipliers, _ = spla.eigs(operator, k=search_k, which="LM", tol=1.0e-8)
        dominant = multipliers[int(np.argmax(np.abs(multipliers)))]
        exponent = np.log(complex(dominant)) / PERIOD
        unstable = int(np.count_nonzero(np.abs(multipliers) > 1.0 + 1.0e-6))
        rows.append((label, dominant, exponent, unstable))
        print(
            f"{label}: |mu|={abs(dominant):.9f}, "
            f"Re(lambda)={exponent.real:+.6e}, unstable={unstable}"
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("candidate", "mu_real", "mu_imag", "mu_abs", "lambda_real", "lambda_imag", "unstable"))
        for label, multiplier, exponent, unstable in rows:
            writer.writerow((label, multiplier.real, multiplier.imag, abs(multiplier), exponent.real, exponent.imag, unstable))
    print(f"csv={OUTPUT}")


if __name__ == "__main__":
    main()
