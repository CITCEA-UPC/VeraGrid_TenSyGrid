# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
EMT Floquet Linear Operators Module.

This module provides high-performance LinearOperator subclasses compatible with
SciPy's sparse linear algebra solvers (like ARPACK). These operators define the
action of the Monodromy Matrix (state-transition over one full limit cycle)
without explicitly constructing it.

Included Operators:
- EmtFloquetOperator: Standard (vector-by-vector) LU-cached operator.
- BlockEmtFloquetOperator: Block (matrix-by-matrix) LU-cached operator.
- AkStackBlockEmtFloquetOperator: Explicit pre-computed transition matrix stack operator.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Any, Optional
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_floquet_numba_kernels import (NUMBA_AVAILABLE,
                                                                                          apply_ak_stack_block_numba)
from VeraGridEngine.basic_structures import Mat, Vec
from VeraGridEngine.enumerations import DynamicIntegrationMethod


def _differential_base_indices(problem: Any) -> np.ndarray:
    """Return the state-vector column owned by each differential variable."""
    state_index = {var.uid: i for i, var in enumerate(problem.get_state_vars())}
    return np.asarray([
        state_index.get(diff_var.base_var.uid, -1) if diff_var.base_var is not None else -1
        for diff_var in problem.get_diff_vars()
    ], dtype=np.int64)


def _structural_mass_matrices(n_total: int, n_states: int,
                              diff_base_indices: np.ndarray) -> tuple[sp.csc_matrix, sp.csc_matrix]:
    """Build derivative-to-residual and state-history mass mappings."""
    n_diff = len(diff_base_indices)
    diff_rows = np.arange(n_diff, dtype=np.int64)
    mass_derivative = sp.csc_matrix(
        (np.ones(n_diff), (diff_rows, diff_rows)), shape=(n_total, n_diff)
    )
    valid = diff_base_indices >= 0
    mass_state = sp.csc_matrix(
        (np.ones(int(np.count_nonzero(valid))),
         (diff_rows[valid], diff_base_indices[valid])),
        shape=(n_total, n_states),
    )
    return mass_state, mass_derivative


class EmtFloquetOperator(spla.LinearOperator):
    """
    HPC Matrix-Free Monodromy Operator for abc-frame EMT systems.

    Optimized with LU Factorization Caching to eliminate redundant computations
    during Krylov subspace iterations (Arnoldi). This operator processes single
    perturbation vectors.
    """
    __slots__ = (
        "problem",
        "trajectory",
        "h",
        "n_states",
        "method",
        "n_total",
        "lu_solvers",
    )

    def __init__(self, problem: Any, trajectory: Mat, h: float, n_states: int,
                 method: DynamicIntegrationMethod = DynamicIntegrationMethod.DaeBackEuler,
                 jac_evaluator: Optional[Any] = None, static_params:Optional[Vec] = None, n_event_params: int = 0,
                 t_trajectory: Optional[Vec] = None, d_trajectory: Optional[Mat] = None):

        """
        Initializes the standard EmtFloquetOperator.

        Args:
            problem: The underlying EMT DAE problem instance.
            trajectory: Pre-captured steady-state trajectory over one period.
            h: Integration time step.
            n_states: Number of differential state variables.
            method_str: Numerical integration method (e.g., 'back_euler').
            jac_evaluator: JIT-compiled Jacobian evaluator function.
            static_params: Array of static system parameters.
            n_event_params: Number of event-driven parameters.
            t_trajectory: Time array corresponding to the limit cycle trajectory.
        """

        self.n_diff = len(problem.get_diff_vars())
        self.diff_base_indices = _differential_base_indices(problem)
        operator_size = n_states + self.n_diff if method == DynamicIntegrationMethod.DaeTrapezoidal else n_states
        op_shape = (operator_size, operator_size)
        op_dtype = np.float64

        super().__init__(dtype=op_dtype, shape=op_shape)

        self.problem = problem
        self.trajectory = trajectory
        self.h = h
        self.n_states = n_states
        self.method = method

        self.n_total = self.problem.get_states_number() + self.problem.get_algebraic_var_number()

        self.lu_solvers = []
        self.mass_state_matrices = []
        self.mass_derivative_matrices = []
        self.mass_state, self.mass_derivative = _structural_mass_matrices(
            self.n_total, self.n_states, self.diff_base_indices
        )

        # Pre-computing Router
        if jac_evaluator is not None and t_trajectory is not None:
            self._precompute_from_jit(jac_evaluator, static_params, n_event_params, t_trajectory, d_trajectory)
        else:
            self._precompute_lu_factorizations()

    def _precompute_from_jit(self, jac_evaluator: Any, static_params: Vec, n_ev_params: int,
                             t_trajectory: Vec, d_trajectory: Optional[Mat])->None:
        """
        Evaluates and caches sparse LU factorizations using the JIT compiled evaluator
        :param jac_evaluator:
        :param static_params:
        :param n_ev_params:
        :param t_trajectory:
        :return:
        """

        full_params = np.empty(n_ev_params + len(static_params), dtype=np.float64)
        if len(static_params) > 0:
            full_params[n_ev_params:] = static_params

        ev_params = np.zeros(n_ev_params, dtype=np.float64)
        dx_dummy = np.zeros(self.n_total, dtype=np.float64)
        if self.method == DynamicIntegrationMethod.DaeTrapezoidal and d_trajectory is None:
            raise ValueError("Trapezoidal Floquet requires the captured derivative trajectory")

        for i in range(1, len(self.trajectory)):
            step_k = self.trajectory[i]
            step_prev = self.trajectory[i - 1]
            step_prev2 = self.trajectory[i - 2] if i > 1 else step_prev

            x_k = step_k[0] if isinstance(step_k, tuple) else step_k
            x_prev = step_prev[0] if isinstance(step_prev, tuple) else step_prev
            x_prev2 = step_prev2[0] if isinstance(step_prev2, tuple) else step_prev2
            t_curr = t_trajectory[i] if t_trajectory is not None else (i * self.h)

            if n_ev_params > 0:
                ev_params = self.problem.def_event_params_fn(ev_params, float(t_curr))
                full_params[:n_ev_params] = ev_params

            d_prev = dx_dummy.copy()
            if d_trajectory is not None:
                d_prev[:self.n_states] = np.asarray(d_trajectory[i - 1], dtype=np.float64)[:self.n_states]
            if self.method == DynamicIntegrationMethod.DaeTrapezoidal:
                J_sparse = jac_evaluator(states=x_k, params=full_params, history=x_prev,
                                         d_history=d_prev, h=self.h, history2=x_prev2)
                self.mass_state_matrices.append(self.mass_state)
                self.mass_derivative_matrices.append(self.mass_derivative)
            else:
                J_sparse = jac_evaluator(states=x_k, params=full_params, history=x_prev,
                                         d_history=dx_dummy, h=self.h, history2=x_prev2)
            self.lu_solvers.append(spla.splu(J_sparse))

    def _precompute_lu_factorizations(self)-> None:
        """
        Pre-calculates SuperLU factorizations for the entire periodic trajectory.
        This reduces the cost of Arnoldi iterations from O(iters * steps * n^3)
        to O(steps * n^3 + iters * steps * n^2), an extreme HPC acceleration.
        """
        for x_k, y_k in self.trajectory:
            fx = self.problem.get_j11(x_k, y_k, self.h)
            fy = self.problem.get_j12(x_k, y_k, self.h)
            gx = self.problem.get_j21(x_k, y_k, self.h)
            gy = self.problem.get_j22(x_k, y_k, self.h)

            J_aug = sp.vstack([
                sp.hstack([fx, fy]),
                sp.hstack([gx, gy])
            ], format='csc')

            # splu requires CSC format and computes the exact LU factorization once
            self.lu_solvers.append(spla.splu(J_aug))

    def _matvec(self, v0: Vec)-> Vec:
        """
        Applies the Monodromy operator to a single vector v0
        :param v0:
        :return:
        """
        v_curr = np.zeros(self.n_total, dtype=np.float64)
        v_curr[:self.n_states] = v0[:self.n_states]

        v_prev = v_curr.copy()
        v_prev2 = v_curr.copy()
        # Store eta = h * delta(xdot) in the augmented coordinate.  This is a
        # similarity scaling of the discrete map and avoids a 1/h imbalance in
        # Arnoldi between state and derivative-history components.
        eta_prev = np.zeros(self.n_diff, dtype=np.float64)
        if self.method == DynamicIntegrationMethod.DaeTrapezoidal:
            eta_prev[:] = v0[self.n_states:]
        rhs = np.zeros_like(v_curr)

        for step_index, lu in enumerate(self.lu_solvers):
            if self.method == DynamicIntegrationMethod.DaeBackEuler:
                rhs[:self.n_states] = v_prev[:self.n_states] / self.h
            elif self.method == DynamicIntegrationMethod.DaeBDF2:
                rhs[:self.n_states] = (2.0 * v_prev[:self.n_states] - 0.5 * v_prev2[:self.n_states]) / self.h
            elif self.method == DynamicIntegrationMethod.DaeTrapezoidal:
                rhs[:] = (
                    (2.0 / self.h) * (self.mass_state_matrices[step_index] @ v_prev[:self.n_states])
                    + (1.0 / self.h) * (self.mass_derivative_matrices[step_index] @ eta_prev)
                )
            else:
                rhs[:self.n_states] = v_prev[:self.n_states] / self.h

            v_curr = lu.solve(rhs)

            if self.method == DynamicIntegrationMethod.DaeTrapezoidal:
                for diff_index, base_index in enumerate(self.diff_base_indices):
                    if base_index >= 0:
                        eta_prev[diff_index] = 2.0 * (
                            v_curr[base_index] - v_prev[base_index]
                        ) - eta_prev[diff_index]

            v_prev2 = v_prev.copy()
            v_prev = v_curr.copy()

        if self.method == DynamicIntegrationMethod.DaeTrapezoidal:
            return np.concatenate((v_curr[:self.n_states], eta_prev))
        return v_curr[:self.n_states]

    def _rmatvec(self, w0: Vec) -> Vec:
        """
        Adjoint (Backward) integration to extract Left Eigenvectors.
        Utilizes the trans='T' flag in SuperLU for zero-cost matrix transposition.
        :param w0:
        :return:
        """
        w_curr = w0.copy()
        rhs = np.zeros(self.n_total)

        # Reverse-time integration using the transposed solvers
        for lu in reversed(self.lu_solvers):
            rhs[:self.n_states] = w_curr

            # Fast back-substitution solving J_aug^T * sol = rhs
            sol = lu.solve(rhs, trans='T')
            w_curr = sol[:self.n_states]

        return w_curr

#New Implementation System for Blocks calc.
class BlockEmtFloquetOperator(spla.LinearOperator):
    """
    Advanced HPC Block Monodromy Operator for EMT Small-Signal Stability.

    This class is specifically designed to support Randomized Numerical Linear
    Algebra (RandNLA) and Block Krylov subspace methods. Unlike standard operators
    that process a single column vector at a time (BLAS Level 2), this operator
    accepts a dense block matrix (N x p) of vectors simultaneously.

    By processing multiple perturbation vectors at once, it forces the underlying
    SciPy/SuperLU backend to utilize BLAS Level 3 routines (Matrix-Matrix
    multiplication and Block Forward/Backward Substitution). This maximizes CPU
    L1/L2 cache utilization, drastically reducing the memory-bound bottlenecks
    inherent to large-scale sparse electrical networks.

    Attributes:
        problem: The underlying EMT DAE problem instance.
        trajectory: List of pre-captured steady-state tuples (x, y) over one period T.
        h (float): Integration time step.
        n_states (int): Number of differential state variables.
        lu_solvers (list): Cached SuperLU factorizations for the entire trajectory.
    """
    __slots__ = (
        "problem",
        "trajectory",
        "h",
        "n_states",
        "method",
        "n_total",
        "lu_solvers",
    )

    def __init__(self, problem: Any, trajectory: Mat, h: float, n_states: int,
                 method: DynamicIntegrationMethod = DynamicIntegrationMethod.DaeBackEuler,
                 jac_evaluator: Optional[Any]=None, static_params:Optional[Vec]=None, n_event_params:int =0,
                 t_trajectory:Optional[Vec]=None, d_trajectory: Optional[Mat] = None):
        """
        Initializes the BlockEmtFloquetOperator.

        Args:
            problem: The underlying EMT DAE problem instance.
            trajectory: List of pre-captured steady-state tuples (x, y) over one period T.
            h: Integration time step.
            n_states: Number of differential state variables.
            method_str: Numerical integration method string.
            jac_evaluator: JIT-compiled Jacobian evaluator function.
            static_params: Array of static system parameters.
            n_event_params: Number of event-driven parameters.
            t_trajectory: Time array corresponding to the limit cycle trajectory.
        """

        n_total_calc = problem.get_states_number() + problem.get_algebraic_var_number()
        self.n_diff = len(problem.get_diff_vars())
        self.diff_base_indices = _differential_base_indices(problem)
        operator_size = n_states + self.n_diff if method == DynamicIntegrationMethod.DaeTrapezoidal else n_states
        op_shape = (operator_size, operator_size)
        op_dtype = np.float64

        super().__init__(dtype=op_dtype, shape=op_shape)

        self.problem = problem
        self.trajectory = trajectory
        self.h = h
        self.n_states = n_states
        self.method = method

        self.n_total = n_total_calc

        self.lu_solvers = []
        self.mass_state_matrices = []
        self.mass_derivative_matrices = []
        self.mass_state, self.mass_derivative = _structural_mass_matrices(
            self.n_total, self.n_states, self.diff_base_indices
        )

        if jac_evaluator is not None:
            self._precompute_from_jit(jac_evaluator, static_params, n_event_params, t_trajectory, d_trajectory)
        else:
            self._precompute_lu_factorizations()

    def _precompute_from_jit(self, jac_evaluator: Any, static_params: Vec,
                             n_ev_params:int, t_trajectory:Optional[Vec]=None,
                             d_trajectory: Optional[Mat] = None)->None:
        """
        Evaluates and caches sparse LU factorizations using the JIT compiled evaluator.
        :param jac_evaluator:
        :param static_params:
        :param n_ev_params:
        :param t_trajectory:
        :return:
        """
        full_params = np.empty(n_ev_params + len(static_params), dtype=np.float64)
        if len(static_params) > 0:
            full_params[n_ev_params:] = static_params

        ev_params = np.zeros(n_ev_params, dtype=np.float64)
        dx_dummy = np.zeros(self.n_total, dtype=np.float64)
        if self.method == DynamicIntegrationMethod.DaeTrapezoidal and d_trajectory is None:
            raise ValueError("Trapezoidal Floquet requires the captured derivative trajectory")

        for i in range(1, len(self.trajectory)):
            step_k = self.trajectory[i]
            step_prev = self.trajectory[i - 1]
            step_prev2 = self.trajectory[i - 2] if i > 1 else step_prev

            x_k = step_k[0] if isinstance(step_k, tuple) else step_k
            x_prev = step_prev[0] if isinstance(step_prev, tuple) else step_prev
            x_prev2 = step_prev2[0] if isinstance(step_prev2, tuple) else step_prev2
            t_curr = t_trajectory[i] if t_trajectory is not None else (i * self.h)

            if n_ev_params > 0:
                ev_params = self.problem.def_event_params_fn(ev_params, float(t_curr))
                full_params[:n_ev_params] = ev_params

            d_prev = dx_dummy.copy()
            if d_trajectory is not None:
                d_prev[:self.n_states] = np.asarray(d_trajectory[i - 1], dtype=np.float64)[:self.n_states]
            if self.method == DynamicIntegrationMethod.DaeTrapezoidal:
                J_sparse = jac_evaluator(states=x_k, params=full_params, history=x_prev,
                                         d_history=d_prev, h=self.h, history2=x_prev2)
                self.mass_state_matrices.append(self.mass_state)
                self.mass_derivative_matrices.append(self.mass_derivative)
            else:
                J_sparse = jac_evaluator(states=x_k, params=full_params, history=x_prev,
                                         d_history=dx_dummy, h=self.h, history2=x_prev2)

            self.lu_solvers.append(spla.splu(J_sparse))

    def _precompute_lu_factorizations(self)-> None:
        """
        Pre-calculates and caches the Sparse LU factorizations for the Jacobian
        at every time step of the periodic limit cycle.
        """
        for x_k, y_k in self.trajectory:
            # Extract discrete DAE Jacobians from the exact analytical/AD backend
            fx = self.problem.get_j11(x_k, y_k, self.h)
            fy = self.problem.get_j12(x_k, y_k, self.h)
            gx = self.problem.get_j21(x_k, y_k, self.h)
            gy = self.problem.get_j22(x_k, y_k, self.h)

            # Assemble the full augmented Jacobian block matrix
            J_aug = sp.vstack([
                sp.hstack([fx, fy]),
                sp.hstack([gx, gy])
            ], format='csc')

            # splu requires CSC format and caches the exact LU decomposition
            self.lu_solvers.append(spla.splu(J_aug))

    def _matmat(self, X: Mat) -> Mat:
        """
        Applies the Monodromy operator to a dense block of vectors.

        Args:
            X: Matrix block of shape (n_states, p).

        Returns:
            Propagated block matrix of shape (n_states, p).
        """
        p_cols = X.shape[1]
        X_curr = np.zeros((self.n_total, p_cols), dtype=X.dtype)
        X_curr[:self.n_states, :] = X[:self.n_states, :]

        X_prev = X_curr.copy()
        X_prev2 = X_curr.copy()
        # Scaled derivative history eta = h * delta(xdot).
        eta_prev = np.zeros((self.n_diff, p_cols), dtype=X.dtype)
        if self.method == DynamicIntegrationMethod.DaeTrapezoidal:
            eta_prev[:, :] = X[self.n_states:, :]
        rhs_block = np.zeros_like(X_curr)

        for step_index, lu in enumerate(self.lu_solvers):
            if self.method == DynamicIntegrationMethod.DaeBackEuler:
                rhs_block[:self.n_states, :] = X_prev[:self.n_states, :] / self.h
            elif self.method == DynamicIntegrationMethod.DaeBDF2:
                rhs_block[:self.n_states, :] = (2.0 * X_prev[:self.n_states, :] - 0.5 * X_prev2[
                    :self.n_states, :]) / self.h
            elif self.method == DynamicIntegrationMethod.DaeTrapezoidal:
                rhs_block[:, :] = (
                    (2.0 / self.h) * (self.mass_state_matrices[step_index] @ X_prev[:self.n_states, :])
                    + (1.0 / self.h) * (self.mass_derivative_matrices[step_index] @ eta_prev)
                )
            else:
                rhs_block[:self.n_states, :] = X_prev[:self.n_states, :] / self.h

            X_curr = lu.solve(rhs_block)

            if self.method == DynamicIntegrationMethod.DaeTrapezoidal:
                for diff_index, base_index in enumerate(self.diff_base_indices):
                    if base_index >= 0:
                        eta_prev[diff_index, :] = 2.0 * (
                            X_curr[base_index, :] - X_prev[base_index, :]
                        ) - eta_prev[diff_index, :]

            X_prev2 = X_prev.copy()
            X_prev = X_curr.copy()

        if self.method == DynamicIntegrationMethod.DaeTrapezoidal:
            return np.vstack((X_curr[:self.n_states, :], eta_prev))
        return X_curr[:self.n_states, :]

    def _matvec(self, v0:Vec)-> Vec:
        """
        Fallback routine: If a standard solver (like scipy.sparse.linalg.eigs)
        calls this operator with a single 1D vector, it wraps it into a 2D matrix
        and routes it through the high-performance _matmat implementation.
        """
        # Reshape to (N, 1), compute via block logic, and flatten back to 1D
        return self._matmat(v0.reshape(-1, 1)).flatten()


class AkStackBlockEmtFloquetOperator(spla.LinearOperator):
    """
    Monodromy operator from an explicit stack of state-transition matrices A_k.

    This is the preferred path for EMT/Floquet benchmarking when the EMT engine can
    expose the per-step linearized state map:
        x_{k+1} = A_k x_k
    for one limit-cycle period. The operator applies:
        Phi = A_{M-1} * ... * A_1 * A_0
    on vectors or blocks. When Numba is available and use_numba=True, the block
    path is JIT-compiled to reduce Python overhead in repeated Arnoldi calls.
    """
    __slots__ = (
        "Ak_stack",
        "n_states",
        "n_steps",
        "use_numba",
    )

    def __init__(self, Ak_stack: Mat, use_numba: bool = True, **kwargs):
        """
        Initializes the AkStackBlockEmtFloquetOperator.

        Args:
            Ak_stack: 3D numpy array representing the sequence of transition matrices.
            use_numba: Flag to enable Numba acceleration for the block multiplication.
        """
        Ak = np.asarray(Ak_stack, dtype=np.float64, order='C')
        if Ak.ndim != 3 or Ak.shape[1] != Ak.shape[2]:
            raise ValueError(
                f"Ak_stack must have shape (n_steps, n, n); got {Ak.shape}"
            )
        self.n_states = int(Ak.shape[1])
        self.Ak_stack = Ak
        self.n_steps = int(Ak.shape[0])
        self.use_numba = bool(use_numba and NUMBA_AVAILABLE and apply_ak_stack_block_numba is not None)
        super().__init__(dtype=np.float64, shape=(self.n_states, self.n_states), **kwargs)

    def _matvec(self, x: Vec) -> Vec:
        """Applies the operator to a single 1D vector."""
        X = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        Y = self._matmat(X)
        return Y[:, 0]

    def _matmat(self, X: Mat) -> Mat:
        """Applies the transition matrix stack to a block of vectors."""
        Xf = np.asarray(X, dtype=np.float64, order='C')
        if self.use_numba:
            return apply_ak_stack_block_numba(self.Ak_stack, Xf)

        Y = Xf
        for k in range(self.n_steps):
            Y = self.Ak_stack[k] @ Y
        return np.asarray(Y, dtype=np.float64, order='C')

    def matmat(self, X: Mat) -> Mat:
        """Public alias for matrix-matrix multiplication."""
        return self._matmat(X)
