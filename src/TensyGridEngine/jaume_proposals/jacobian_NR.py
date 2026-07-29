"""
Newton-Raphson solver for the CPN1 steady-state system.

Solves: Phi @ varphi(v) = 0

where varphi_j(v) = prod_p (s_{i_p,j} * v_{i_p} + (1 - |s_{i_p,j}|))

Unlike power flow which solves P(V,θ)=0, Q(V,θ)=0 for bus voltages only,
this solver finds the full steady-state including generator controller states
(exciter voltages, governor positions, etc.) by setting all time derivatives
to zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

project_base = Path(__file__).resolve().parents[3]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from TensyGridEngine.cpn_utils import build_problems, process_cpn_system, compute_jacobian_from_S


def residual(S: sparse.csc_matrix, Phi: sparse.csr_matrix, v: np.ndarray) -> np.ndarray:
    """Compute the residual vector f(v) = Phi @ varphi(v).

    Args:
        S: Variable-to-monomial mapping matrix (n_vars, n_mon).
        Phi: Equation coefficients matrix (n_eqs, n_mon).
        v: Current system variable values (n_vars,).

    Returns:
        f: Residual vector (n_eqs,).
    """
    data = S.data
    indices = S.indices
    indptr = S.indptr
    n_mon = S.shape[1]
    varphi = np.empty(n_mon)

    for j in range(n_mon):
        start = indptr[j]
        end = indptr[j + 1]
        prod = 1.0
        for p in range(start, end):
            i = indices[p]
            s_ij = data[p]
            prod *= (s_ij * v[i] + (1.0 - abs(s_ij)))
        varphi[j] = prod

    return Phi @ varphi


def equilibrate_rows_csr(Phi: sparse.csr_matrix) -> sparse.csr_matrix:
    """Equilibra las filas de Phi dividiendo cada fila por su elemento de máximo valor absoluto."""
    Phi = Phi.tocsr().copy()
    n_rows = Phi.shape[0]
    row_max = np.zeros(n_rows, dtype=np.float64)

    # Extracción directa del valor máximo por fila usando la estructura CSR
    for i in range(n_rows):
        start = Phi.indptr[i]
        end = Phi.indptr[i + 1]
        if start < end:
            row_max[i] = np.max(np.abs(Phi.data[start:end]))

    # Evitar división por cero en filas completamente vacías/nulas
    row_max[row_max == 0.0] = 1.0

    # Aplicar el escalado mediante multiplicación por matriz diagonal dispersa
    inv_max = 1.0 / row_max
    D = sparse.diags(inv_max)
    return D @ Phi

def jacobian_dense(S: sparse.spmatrix, Phi: sparse.spmatrix, v: np.ndarray) -> np.ndarray:
    """Compute the dense system Jacobian J = Phi @ F^T.

    Args:
        S: Variable-to-monomial mapping matrix.
        Phi: Equation coefficients matrix.
        v: Current variable values.

    Returns:
        J: Dense Jacobian matrix (n_eqs, n_vars).
    """
    F = compute_jacobian_from_S(S, v)
    return (Phi @ F.T).toarray()


def newton_direct(
    S: sparse.spmatrix,
    Phi: sparse.spmatrix,
    x0: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-5,
    alpha_min: float = 1e-4,
) -> np.ndarray:
    """Solve the CPN1 system using Newton-Raphson with direct linear solve
    and backtracking line search for globalization.

    Uses np.linalg.solve when J is square and well-conditioned, falling back
    to lstsq for non-square (n_eqs != n_vars) or singular systems.

    Args:
        S: Variable-to-monomial mapping matrix.
        Phi: Equation coefficients matrix.
        x0: Initial state vector guess.
        max_iter: Maximum number of iterations allowed.
        tol: Convergence tolerance on residual norm ||f||.
        alpha_min: Minimum step size scaling factor for backtracking line search.

    Returns:
        v: Converged or final solution vector.
    """
    v = x0.copy()
    Phi = equilibrate_rows_csr(Phi)
    f = residual(S, Phi, v)
    err = np.linalg.norm(f)
    n_eqs, n_vars = Phi.shape[0], S.shape[0]
    print(f"iter 0: ||f|| = {err:.6e}, n_eqs={n_eqs}, n_vars={n_vars}")

    if err < tol:
        return v

    for k in range(1, max_iter + 1):
        J = jacobian_dense(S, Phi, v)

        # Attempt direct square solve if dimensions match, else use least-squares
        if n_eqs == n_vars:
            try:
                dx = np.linalg.solve(J, f)
            except (np.linalg.LinAlgError, ValueError):
                dx, _, _, _ = np.linalg.lstsq(J, f, rcond=None)
        else:
            dx, _, _, _ = np.linalg.lstsq(J, f, rcond=None)

        # Backtracking line search to guarantee residual reduction
        alpha = 1.0
        v_candidate = v - alpha * dx
        f_candidate = residual(S, Phi, v_candidate)
        err_candidate = np.linalg.norm(f_candidate)

        while err_candidate >= err and alpha > alpha_min:
            alpha *= 0.5
            v_candidate = v - alpha * dx
            f_candidate = residual(S, Phi, v_candidate)
            err_candidate = np.linalg.norm(f_candidate)

        if err_candidate >= err:
            dx, _, _, _ = np.linalg.lstsq(J, f, rcond=None)
            alpha = 1.0
            v_candidate = v - alpha * dx
            f_candidate = residual(S, Phi, v_candidate)
            err_candidate = np.linalg.norm(f_candidate)
            while err_candidate >= err and alpha > alpha_min:
                alpha *= 0.5
                v_candidate = v - alpha * dx
                f_candidate = residual(S, Phi, v_candidate)
                err_candidate = np.linalg.norm(f_candidate)

        if err_candidate >= err:
            print(f"iter {k}: no descent direction found (||f|| = {err:.6e}), stopping.")
            break

        v_new = v_candidate
        f_new = f_candidate
        err_new = err_candidate

        if np.any(np.isnan(v_new)) or np.any(np.isinf(v_new)):
            print(f"iter {k}: NaN/Inf encountered during step update.")
            break

        v = v_new
        f = f_new
        err = err_new

        if k <= 10 or k % 20 == 0 or err < 1e-4:
            print(f"iter {k} (alpha={alpha:.3f}): ||f|| = {err:.6e}")

        if err < tol:
            print(f"Convergence achieved at iteration {k}.")
            break

    return v


def main() -> None:
    grid_filename = sys.argv[1] if len(sys.argv) > 1 else "IEEE 9 Bus.gridcal"
    problem_ml, pf_results, rms_options_ml = build_problems(grid_filename=grid_filename)
    problem_ml.build_multilinear_matrices()

    cpn = process_cpn_system(problem_ml, jaume_flag=True)

    n_vars = cpn.S.shape[0]
    n_eqs = cpn.Phi.shape[0]
    n_mon = cpn.S.shape[1]
    print(f"S shape: {cpn.S.shape}, Phi shape: {cpn.Phi.shape}")

    x0 = cpn.x0[cpn.orig_keep_idx].copy()

    J = jacobian_dense(cpn.S, cpn.Phi, x0)
    print("\n=== Jacobian Statistics at x0 ===")
    print(f"  Shape: {J.shape}")
    print(f"  nnz: {np.count_nonzero(J)}, density: {np.count_nonzero(J) / J.size:.4f}")
    print(f"  ||J||_F   = {np.linalg.norm(J, 'fro'):.6e}")
    print(f"  ||J||_1   = {np.linalg.norm(J, 1):.6e}")
    print(f"  ||J||_inf = {np.linalg.norm(J, np.inf):.6e}")

    U, s, Vt = np.linalg.svd(J, full_matrices=True)
    rank = int(np.sum(s > 1e-10 * s[0])) if s[0] > 0 else 0
    print(f"  Singular values (total {len(s)}):")
    print(f"    s[0]   = {s[0]:.6e}")
    print(f"    s[mid] = {s[len(s)//2]:.6e}")
    print(f"    s[-1]  = {s[-1]:.6e}")
    print(f"    s[-5:] = {s[-5:]}")

    if 0 < rank < len(s):
        print(f"    s[{rank-1}] = {s[rank-1]:.6e} (last value above tolerance)")
        print(f"    s[{rank}]   = {s[rank]:.6e} (first value below tolerance)")

    cond = s[0] / s[rank - 1] if rank > 0 and s[rank - 1] > 0 else np.inf
    print(f"  Rank: {rank}/{J.shape[0]}")
    print(f"  kappa(J) = {cond:.6e}")

    if n_eqs == n_vars:
        JtJ = J.T @ J
        eigvals = np.linalg.eigvalsh(JtJ)
        num_neg_eigs = np.sum(eigvals < 0.0)  # Correctly count negative eigenvalues before clamping
        num_zero_eigs = np.sum(np.abs(eigvals) < 1e-12)
        eigvals_clamped = np.maximum(eigvals, 0.0)

        print("\n  Eigenvalues of J^T J:")
        print(f"    max = {eigvals_clamped[-1]:.6e}")
        print(f"    min = {eigvals_clamped[0]:.6e}")
        print(f"    condition number = {eigvals_clamped[-1] / max(eigvals_clamped[0], 1e-30):.6e}")
        print(f"    negative counts: {num_neg_eigs}")
        print(f"    near-zero (<1e-12): {num_zero_eigs}")

    f0 = residual(cpn.S, cpn.Phi, x0)
    print(f"\n  ||f(x0)||_2   = {np.linalg.norm(f0):.6e}")
    print(f"  ||f(x0)||_inf = {np.max(np.abs(f0)):.6e}")

    print("\n--- Running Newton-Raphson Solver ---")
    v_newton = newton_direct(cpn.S, cpn.Phi, x0)
    f_newton = residual(cpn.S, cpn.Phi, v_newton)
    print(f"\nFinal ||f||_2 = {np.linalg.norm(f_newton):.6e}")
    print(f"Max |f_i|     = {np.max(np.abs(f_newton)):.6e}")


if __name__ == "__main__":
    main()