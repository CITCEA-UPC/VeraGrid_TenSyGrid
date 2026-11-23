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


def _constraint_basis(constraints: np.ndarray, n_state: int) -> np.ndarray:
    """Return orthonormal state directions satisfying linear constraints."""
    if constraints.shape[0]:
        _, singular_values, vh = la.svd(constraints, full_matrices=True)
        rank = _numerical_rank(singular_values, constraints.shape)
        return vh[rank:, :].T.copy()
    return np.eye(n_state)


def _sparse_algebraic_tangent(gx: sp.csc_matrix, gz: sp.csc_matrix
                              ) -> tuple[np.ndarray, np.ndarray] | None:
    """Eliminate a matched sparse minor, preserving minimum-norm sensitivities.

    Structural matching exposes the small null-space problem without taking
    an SVD of the entire network algebraic Jacobian. Numerically singular
    matched minors fall back to the general dense rank-revealing calculation.
    """
    gz = gz.copy()
    gz.eliminate_zeros()
    matching = sp.csgraph.maximum_bipartite_matching(gz, perm_type="column")
    rows = np.flatnonzero(matching >= 0)
    columns = matching[rows]
    remaining_rows = np.setdiff1d(np.arange(gz.shape[0]), rows)
    remaining_columns = np.setdiff1d(np.arange(gz.shape[1]), columns)
    if not len(rows) or len(remaining_columns) > gz.shape[1] // 2:
        return None
    try:
        factor = spla.splu(gz[rows][:, columns].tocsc())
        # Left null vectors are [-W, I]. Whiten them so the numerical rank
        # decision has the same orthonormal scaling as the dense SVD path.
        coupling = gz[remaining_rows][:, columns].toarray()
        weights = factor.solve(coupling.T, trans="T").T
        free_block = gz[rows][:, remaining_columns].toarray()
        remainder = gz[remaining_rows][:, remaining_columns].toarray() - weights @ free_block
        if remainder.size and np.max(np.abs(remainder)) > 1e-8:
            return None
        hidden = gx[remaining_rows].toarray() - weights @ gx[rows]
        if len(remaining_rows):
            left_gram = np.eye(len(remaining_rows)) + weights @ weights.T
            chol = la.cholesky(left_gram, lower=True)
            hidden = la.solve_triangular(chol, hidden, lower=True)
        basis = _constraint_basis(np.asarray(hidden), gx.shape[1])
        particular = -factor.solve(np.asarray(gx[rows] @ basis))
        # z_dep = particular + free_sensitivity*z_free. Select the same
        # minimum-norm algebraic perturbation as the Moore-Penrose inverse.
        free_sensitivity = -factor.solve(free_block)
        if len(remaining_columns):
            right_gram = np.eye(len(remaining_columns)) + free_sensitivity.T @ free_sensitivity
            free_values = -la.solve(right_gram, free_sensitivity.T @ particular, assume_a="pos")
            particular += free_sensitivity @ free_values
        else:
            free_values = np.empty((0, basis.shape[1]))
        algebraic = np.empty((gz.shape[1], basis.shape[1]))
        algebraic[columns] = particular
        algebraic[remaining_columns] = free_values
        residual = np.asarray(gx @ basis + gz @ algebraic)
        if residual.size and np.linalg.norm(residual, ord=np.inf) > 1e-7:
            return None
        return basis, algebraic
    except (RuntimeError, la.LinAlgError):
        return None


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
    continuous_jacobian = (jacobian - (2.0 / h) * discrete_mass).tocsc()
    gx_sparse = continuous_jacobian[n_diff:, :n_state].tocsc()
    gz_sparse = continuous_jacobian[n_diff:, n_state:].tocsc()
    sparse_tangent = _sparse_algebraic_tangent(gx_sparse, gz_sparse) if n_alg >= 256 else None
    if sparse_tangent is not None:
        basis, algebraic = sparse_tangent
        derivative = -(continuous_jacobian[:n_diff, :n_state] @ basis
                       + continuous_jacobian[:n_diff, n_state:] @ algebraic)
        return ConsistentEmtTangent(basis=basis, initial_derivative_map=derivative @ basis.T)
    gx = gx_sparse.toarray()
    gz = gz_sparse.toarray()

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
