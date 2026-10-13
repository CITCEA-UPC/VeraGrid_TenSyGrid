"""Probe local index reduction of the IEEE-9 dynamic pi-line DAE.

The pi-line template has charge states constrained to bus voltages. Direct
algebraic elimination is singular. This experiment differentiates the linear
state constraints and builds a *local numerical* state-only vector field on
their constraint manifold. The resulting solve may depend on the state and
is not claimed to be a multilinear symbolic ODE.
"""

from __future__ import annotations

import argparse
import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp

from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_gfl_ode_reduced import build_problem
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver


def coefficient_data(problem):
    """Split multilinear S/Phi into linear algebraic and state RHS terms."""
    n = len(problem.get_state_vars())
    m = len(problem.get_algebraic_vars())
    S, Phi = problem.build_multilinear_matrices_fast()
    monomials = [tuple(int(i) for i in S.indices[S.indptr[j]:S.indptr[j + 1]])
                 for j in range(S.shape[1])]
    algebraic = Phi[n:].tocsr()
    A = np.zeros((m, m))
    B = np.zeros((m, n))
    constant = np.zeros(m)
    for row in range(m):
        for pointer in range(algebraic.indptr[row], algebraic.indptr[row + 1]):
            term = monomials[algebraic.indices[pointer]]
            coefficient = algebraic.data[pointer]
            if not term:
                constant[row] += coefficient
            elif len(term) == 1 and term[0] < n:
                B[row, term[0]] += coefficient
            elif len(term) == 1 and term[0] < n + m:
                A[row, term[0] - n] += coefficient
            else:
                raise ValueError(f"Nonlinear or differential algebraic term {term}")

    state = Phi[:n].tocsr()
    rhs_terms = []
    for row in range(n):
        for pointer in range(state.indptr[row], state.indptr[row + 1]):
            term = monomials[state.indices[pointer]]
            algebraic_modes = [i - n for i in term if n <= i < n + m]
            if len(algebraic_modes) > 1 or any(i >= n + m for i in term):
                raise ValueError(f"State RHS has unsupported term {term}")
            state_modes = tuple(i for i in term if i < n)
            rhs_terms.append((row, algebraic_modes[0] if algebraic_modes else None,
                              state_modes, float(state.data[pointer])))

    def state_coefficients(x):
        f0 = np.zeros(n)
        Fy = np.zeros((n, m))
        for row, algebraic_mode, state_modes, coefficient in rhs_terms:
            value = coefficient
            for mode in state_modes:
                value *= x[mode]
            if algebraic_mode is None:
                f0[row] += value
            else:
                Fy[row, algebraic_mode] += value
        return f0, Fy

    return A, B, constant, state_coefficients


def main(*, step: float = 2.5e-6) -> None:
    full = build_problem(use_pi_lines=True)
    n = len(full.get_state_vars())
    m = len(full.get_algebraic_vars())
    options = full.options
    options.time_step = step
    solver = build_emt_solver(options=options, problem=full, t0=0.0,
                              t_end=1e-4, h=float(options.time_step),
                              method=options.integration_method)
    times, values, _, initialized, converged = solver.simulate(boundary_updater=full)
    if not (initialized and converged):
        raise RuntimeError("The full pi-line DAE simulation failed")
    # The DAE solver's reported t=0 point is its raw initializer; the first
    # accepted step projects it onto the algebraic constraint manifold.
    start_sample = 1
    x0 = values[start_sample, :n]
    A, B, constant, state_coefficients = coefficient_data(full)
    f0_at_start, Fy_at_start = state_coefficients(x0)

    # Left-null vectors of A give the state constraints hidden in A*y+B*x+c=0.
    first_left_null = linalg.null_space(A.T)
    first_constraints = first_left_null.T @ B
    first_differentiated = first_constraints @ Fy_at_start
    if np.linalg.norm(first_constraints @ f0_at_start) > 1e-8:
        raise RuntimeError("First differentiated constraints have an unhandled state-only term")
    first_augmented = np.vstack((A, first_differentiated))

    # Applying the same procedure after one differentiation exposes six more
    # independent current-balance constraints at buses without line charging.
    second_left_null = linalg.null_space(first_augmented.T)
    second_B = np.vstack((B, np.zeros((first_differentiated.shape[0], n))))
    all_constraints = second_left_null.T @ second_B
    constraint_basis = linalg.orth(all_constraints.T).T
    independent_basis = linalg.null_space(constraint_basis)

    def algebraic_system(x):
        f0, Fy = state_coefficients(x)
        matrix = np.vstack((A, constraint_basis @ Fy))
        rhs = np.r_[-(B @ x + constant), -constraint_basis @ f0]
        return matrix, rhs, f0, Fy

    matrix0, rhs0, _, _ = algebraic_system(x0)
    # Select a nonsingular square subset once. Verify the complete overdetermined
    # system at every evaluation; discarded rows must remain redundant.
    _, _, pivots = linalg.qr(matrix0.T, pivoting=True)
    selected_rows = pivots[:m]
    dae_y0 = values[start_sample, n:n + m]
    initial_algebraic_residual = np.linalg.norm(A @ dae_y0 + B @ x0 + constant)
    initial_tangent_residual = np.linalg.norm(
        constraint_basis @ (f0_at_start + Fy_at_start @ dae_y0)
    )

    def explicit_rhs(_, coordinates):
        x = x0 + independent_basis @ coordinates
        matrix, rhs, f0, Fy = algebraic_system(x)
        y = linalg.solve(matrix[selected_rows], rhs[selected_rows], assume_a="gen")
        residual = np.linalg.norm(matrix @ y - rhs) / max(np.linalg.norm(rhs), 1.0)
        if residual > 1e-7:
            raise RuntimeError(f"Index-reduced algebraic residual is {residual:.3e}")
        return independent_basis.T @ (f0 + Fy @ y)

    initial_coordinates = np.zeros(independent_basis.shape[1])
    solution = solve_ivp(explicit_rhs, (times[start_sample], times[-1]),
                         initial_coordinates, t_eval=times[start_sample:],
                         rtol=1e-9, atol=1e-11)
    if not solution.success:
        raise RuntimeError(solution.message)
    explicit_states = x0[:, None] + independent_basis @ solution.y
    state_error = np.max(np.abs(explicit_states.T - values[start_sample:, :n]))
    state_scale = max(np.max(np.abs(values[start_sample:, :n])), 1.0)
    constraint_error = np.linalg.norm(constraint_basis @ (values[-1, :n] - x0))
    matrix_end, _, _, _ = algebraic_system(values[-1, :n])
    solve_matrix_change = np.linalg.norm(matrix_end - matrix0) / np.linalg.norm(matrix0)
    print(f"full pi-line DAE: states={n}, algebraics={m}, compared samples={len(solution.t)}")
    print(f"algebraic ranks: original={np.linalg.matrix_rank(A)}, "
          f"after first differentiation={np.linalg.matrix_rank(first_augmented)}, "
          f"after second={np.linalg.matrix_rank(matrix0)}")
    print(f"independent state coordinates={independent_basis.shape[1]}")
    print(f"max state difference vs DAE={state_error:.3e} "
          f"(relative to max state magnitude={state_error / state_scale:.3e})")
    print(f"DAE final constraint drift={constraint_error:.3e}")
    print(f"algebraic solve matrix relative change={solve_matrix_change:.3e}")
    print(f"DAE start algebraic residual={initial_algebraic_residual:.3e}, "
          f"tangent residual={initial_tangent_residual:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=2.5e-6,
                        help="DAE trapezoidal step in seconds (default: 2.5e-6)")
    main(step=parser.parse_args().step)
