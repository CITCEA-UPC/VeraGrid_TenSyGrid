"""Trajectory-dependent manifold reduction for EMT Floquet operators."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_floquet_operator import EmtFloquetOperator
from VeraGridEngine.basic_structures import Mat, Vec
from VeraGridEngine.enumerations import DynamicIntegrationMethod


class TrigManifoldReducedEmtFloquetOperator(EmtFloquetOperator):
    """Floquet operator with Sauer--Pai sine/cosine lifts eliminated per step.

    For each ``(theta, u_cos, u_sin)`` state-index triplet, perturbations obey

    ``du_cos = -u_sin(t) dtheta`` and ``du_sin = u_cos(t) dtheta``.

    The two lifted-state residual rows are removed and the relations are
    substituted into every discrete DAE Jacobian before its LU factorization.
    """

    def __init__(
        self,
        problem: Any,
        trajectory: Mat,
        h: float,
        n_states: int,
        trig_triplets: Sequence[tuple[int, int, int]],
        method: DynamicIntegrationMethod = DynamicIntegrationMethod.DaeTrapezoidal,
        jac_evaluator: Optional[Any] = None,
        static_params: Optional[Vec] = None,
        n_event_params: int = 0,
        t_trajectory: Optional[Vec] = None,
    ) -> None:
        if jac_evaluator is None or t_trajectory is None:
            raise ValueError("The reduced operator requires a trajectory Jacobian evaluator and time samples")
        if method not in (
            DynamicIntegrationMethod.DaeBackEuler,
            DynamicIntegrationMethod.DaeBDF2,
            DynamicIntegrationMethod.DaeTrapezoidal,
        ):
            raise ValueError(f"Unsupported integration method: {method}")

        self.problem = problem
        self.trajectory = trajectory
        self.h = h
        self.n_states_full = n_states
        self.n_total_full = n_states + problem.get_algebraic_var_number()
        self.method = method
        self.trig_triplets = tuple(trig_triplets)

        removed = sorted(index for _, cos_index, sin_index in self.trig_triplets
                         for index in (cos_index, sin_index))
        if len(set(removed)) != len(removed):
            raise ValueError("Trig triplets contain repeated lifted-state indices")
        self.kept_state_indices = np.asarray(
            [index for index in range(n_states) if index not in set(removed)], dtype=np.int64
        )
        self.kept_row_indices = np.asarray(
            list(self.kept_state_indices) + list(range(n_states, self.n_total_full)), dtype=np.int64
        )
        self.n_states = len(self.kept_state_indices)
        self.n_total = self.n_states + problem.get_algebraic_var_number()

        spla.LinearOperator.__init__(self, dtype=np.float64, shape=(self.n_states, self.n_states))

        self.injections = [
            self._build_injection(self._state_vector(sample)) for sample in trajectory
        ]
        self.lu_solvers: list[Any] = []
        self._precompute_reduced_factorizations(
            jac_evaluator=jac_evaluator,
            static_params=np.asarray(static_params if static_params is not None else [], dtype=float),
            n_event_params=n_event_params,
            t_trajectory=np.asarray(t_trajectory, dtype=float),
        )

    @staticmethod
    def _state_vector(sample: Any) -> np.ndarray:
        return np.asarray(sample[0] if isinstance(sample, tuple) else sample, dtype=float)

    def _build_injection(self, trajectory_state: np.ndarray) -> sp.csc_matrix:
        """Map reduced differential/algebraic perturbations to full coordinates."""
        state_position = {full: reduced for reduced, full in enumerate(self.kept_state_indices)}
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for reduced, full in enumerate(self.kept_state_indices):
            rows.append(int(full))
            cols.append(reduced)
            data.append(1.0)

        for theta_index, cos_index, sin_index in self.trig_triplets:
            theta_column = state_position[theta_index]
            rows.extend((cos_index, sin_index))
            cols.extend((theta_column, theta_column))
            data.extend((-trajectory_state[sin_index], trajectory_state[cos_index]))

        n_alg = self.problem.get_algebraic_var_number()
        for alg_index in range(n_alg):
            rows.append(self.n_states_full + alg_index)
            cols.append(self.n_states + alg_index)
            data.append(1.0)

        return sp.csc_matrix((data, (rows, cols)), shape=(self.n_total_full, self.n_total))

    def _precompute_reduced_factorizations(
        self,
        jac_evaluator: Any,
        static_params: np.ndarray,
        n_event_params: int,
        t_trajectory: np.ndarray,
    ) -> None:
        full_params = np.empty(n_event_params + len(static_params), dtype=float)
        if len(static_params):
            full_params[n_event_params:] = static_params
        event_params = np.zeros(n_event_params, dtype=float)
        dx_dummy = np.zeros(self.n_total_full, dtype=float)

        for step in range(1, len(self.trajectory)):
            x_k = self._state_vector(self.trajectory[step])
            x_prev = self._state_vector(self.trajectory[step - 1])
            x_prev2 = self._state_vector(self.trajectory[step - 2]) if step > 1 else x_prev
            if n_event_params:
                event_params = self.problem.def_event_params_fn(event_params, float(t_trajectory[step]))
                full_params[:n_event_params] = event_params
            jacobian_full = jac_evaluator(
                states=x_k,
                params=full_params,
                history=x_prev,
                d_history=dx_dummy,
                h=self.h,
                history2=x_prev2,
            )
            jacobian_reduced = jacobian_full[self.kept_row_indices, :] @ self.injections[step]
            if jacobian_reduced.shape != (self.n_total, self.n_total):
                raise RuntimeError(f"Reduced Jacobian is not square: {jacobian_reduced.shape}")
            self.lu_solvers.append(spla.splu(jacobian_reduced.tocsc()))

    def _matvec(self, reduced_initial: Vec) -> Vec:
        full_previous = np.asarray(self.injections[0][:, :self.n_states] @ reduced_initial).ravel()
        full_previous2 = full_previous.copy()
        derivative_previous = np.zeros(self.n_total_full, dtype=float)

        for step, lu in enumerate(self.lu_solvers, start=1):
            rhs_full = np.zeros(self.n_total_full, dtype=float)
            if self.method == DynamicIntegrationMethod.DaeBackEuler:
                rhs_full[:self.n_states_full] = full_previous[:self.n_states_full] / self.h
            elif self.method == DynamicIntegrationMethod.DaeBDF2:
                rhs_full[:self.n_states_full] = (
                    2.0 * full_previous[:self.n_states_full]
                    - 0.5 * full_previous2[:self.n_states_full]
                ) / self.h
            else:
                rhs_full[:self.n_states_full] = (
                    (2.0 / self.h) * full_previous[:self.n_states_full]
                    + derivative_previous[:self.n_states_full]
                )

            reduced_current = lu.solve(rhs_full[self.kept_row_indices])
            full_current = np.asarray(self.injections[step] @ reduced_current).ravel()
            if self.method == DynamicIntegrationMethod.DaeTrapezoidal:
                derivative_previous[:self.n_states_full] = (
                    (2.0 / self.h)
                    * (full_current[:self.n_states_full] - full_previous[:self.n_states_full])
                    - derivative_previous[:self.n_states_full]
                )
            full_previous2 = full_previous
            full_previous = full_current

        return full_previous[self.kept_state_indices]

