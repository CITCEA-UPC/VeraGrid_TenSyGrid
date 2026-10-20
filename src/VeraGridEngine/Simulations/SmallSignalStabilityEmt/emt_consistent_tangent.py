"""Consistent initial tangent coordinates for trapezoidal EMT Floquet analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_floquet_operator import (
    _differential_base_indices,
    _structural_mass_matrices,
)


@dataclass(frozen=True)
class ConsistentEmtTangent:
    """Orthonormal admissible state directions and their derivative sensitivity."""

    basis: np.ndarray
    initial_derivative_map: np.ndarray


class ProjectedEmtFloquetOperator(spla.LinearOperator):
    """Restrict an EMT period map to admissible initial state directions."""

    def __init__(self, operator: spla.LinearOperator, basis: np.ndarray):
        self.operator = operator
        self.basis = np.asarray(basis, dtype=np.float64)
        if self.basis.ndim != 2 or self.basis.shape[0] != operator.shape[0]:
            raise ValueError("Tangent basis has the wrong shape")
        super().__init__(dtype=np.float64, shape=(self.basis.shape[1], self.basis.shape[1]))

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        return self.basis.T @ self.operator.matvec(self.basis @ vector)

    def _matmat(self, vectors: np.ndarray) -> np.ndarray:
        return self.basis.T @ self.operator.matmat(self.basis @ vectors)


def _numerical_rank(values: np.ndarray, shape: tuple[int, int]) -> int:
    if values.size == 0:
        return 0
    return int(np.count_nonzero(values > max(shape) * np.finfo(float).eps * values[0]))


def build_consistent_emt_tangent(
    problem: Any,
    trajectory: np.ndarray,
    derivatives: np.ndarray,
    times: np.ndarray,
    h: float,
    jacobian_evaluator: Any,
    static_parameters: np.ndarray,
    n_event_parameters: int,
) -> ConsistentEmtTangent:
    """Linearize the initial DAE and retain only admissible state perturbations.

    The trapezoidal Jacobian is ``J_continuous + (2/h) M``. Removing the
    discrete mass term exposes the algebraic constraints and the initial
    derivative sensitivity. SVD is used only for this opt-in analysis because
    the algebraic block need not be invertible.
    """
    n_state = problem.get_states_number()
    n_alg = problem.get_algebraic_var_number()
    n_diff = len(problem.get_diff_vars())
    n_total = n_state + n_alg
    if n_diff != n_state:
        raise ValueError("Consistent tangent currently requires one differential per state")
    if h <= 0:
        raise ValueError("Floquet time step must be positive")

    parameters = np.empty(n_event_parameters + len(static_parameters), dtype=float)
    parameters[n_event_parameters:] = static_parameters
    if n_event_parameters:
        parameters[:n_event_parameters] = problem.def_event_params_fn(
            np.zeros(n_event_parameters, dtype=float), float(times[0])
        )
    initial = np.asarray(trajectory[0], dtype=float)
    initial_derivative = np.asarray(derivatives[0], dtype=float)
    jacobian = jacobian_evaluator(
        states=initial, params=parameters, history=initial,
        d_history=initial_derivative, h=h, history2=initial,
    ).tocsc()
    mass_state, _ = _structural_mass_matrices(
        n_total, n_state, _differential_base_indices(problem)
    )
    discrete_mass = sp.hstack(
        (mass_state, sp.csc_matrix((n_total, n_alg))), format="csc"
    )
    continuous_jacobian = (jacobian - (2.0 / h) * discrete_mass).toarray()
    gx = continuous_jacobian[n_diff:, :n_state]
    gz = continuous_jacobian[n_diff:, n_state:]

    if n_alg:
        uz, sz, vzh = la.svd(gz, full_matrices=True)
        rank_z = _numerical_rank(sz, gz.shape)
        hidden_constraints = uz[:, rank_z:].T @ gx
    else:
        uz = np.empty((0, 0))
        sz = np.empty(0)
        vzh = np.empty((0, 0))
        rank_z = 0
        hidden_constraints = np.empty((0, n_state))

    if hidden_constraints.shape[0]:
        _, sh, vh = la.svd(hidden_constraints, full_matrices=True)
        rank_hidden = _numerical_rank(sh, hidden_constraints.shape)
        basis = vh[rank_hidden:, :].T.copy()
    else:
        basis = np.eye(n_state)
    if basis.shape[1] == 0:
        raise ValueError("The initial DAE has no admissible state perturbations")

    if rank_z:
        algebraic = -(vzh[:rank_z, :].T / sz[:rank_z]) @ (uz[:, :rank_z].T @ gx @ basis)
    else:
        algebraic = np.zeros((n_alg, basis.shape[1]))
    residual = gx @ basis + gz @ algebraic
    if residual.size and np.linalg.norm(residual, ord=np.inf) > 1e-7:
        raise ValueError("Initial tangent does not satisfy the algebraic constraints")

    derivative = -(
        continuous_jacobian[:n_diff, :n_state] @ basis
        + continuous_jacobian[:n_diff, n_state:] @ algebraic
    )
    return ConsistentEmtTangent(
        basis=basis,
        initial_derivative_map=derivative @ basis.T,
    )
