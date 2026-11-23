"""Independent representation checks and finite-mode RMS descriptor reduction."""
from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import splu, norm


def check_symbolic_representation(problem, initial):
    """Compare S/Phi against original equations at equilibrium and two perturbations."""
    bindings = {var.uid: float(value) for var, value in
                zip(problem._constant_parameters, problem._constant_params)}
    bindings.update({var.uid: float(value) for var, value in
                     zip(problem._variable_parameters, problem._variable_parameters_values)})
    bindings[problem._glob_time.uid] = 0.
    equations = problem._state_eqs + problem._algebraic_eqs
    rng = np.random.default_rng(17)
    worst = 0.
    worst_scaled = 0.
    symbolic_initial = None
    for sample in range(3):
        values = initial.copy()
        derivatives = np.zeros(problem.get_diff_var_number())
        if sample:
            values += 1e-6*np.maximum(1., abs(values))*rng.normal(size=len(values))
            derivatives = 1e-6*rng.normal(size=len(derivatives))
        bindings.update({uid: float(values[index]) for uid, index in problem._uid2idx_vars.items()})
        bindings.update({var.uid: float(derivatives[i]) for i, var in enumerate(problem._diff_vars)})
        original = np.asarray([float(equation.eval_uid(bindings)) for equation in equations])
        exported = problem.exact_residual_from_s_phi(np.r_[values, derivatives])
        error = float(np.max(abs(original-exported)))
        worst = max(worst, error)
        worst_scaled = max(worst_scaled, error/max(1., float(np.max(abs(original)))))
        if not sample:
            symbolic_initial = float(np.max(abs(original)))
    if not np.isfinite(worst_scaled) or worst_scaled > 1e-10:
        raise RuntimeError(f'S/Phi does not reproduce the original RMS equations: scaled error {worst_scaled}')
    return dict(symbolic_initial_residual_inf=symbolic_initial,
                representation_error_inf=worst, representation_scaled_error=worst_scaled,
                representation_samples=3, representation_validated=True)


def finite_descriptor_modes(problem, initial, derivatives):
    """Eliminate an invertible algebraic block and solve the unchanged finite pencil.

    Returns every finite eigenmode, its reconstructed full right vector and
    normalized residuals against the original descriptor matrices. No diagonal
    regularization, time simulation or target-frequency selection is applied.
    """
    a = sparse.csc_matrix(problem.get_static_state_matrix(initial, derivatives))
    e = -sparse.csc_matrix(problem.get_E_matrix(initial, derivatives))
    e.eliminate_zeros()
    row_indices, column_indices = e.nonzero()
    dynamic_rows = np.unique(row_indices)
    dynamic_columns = np.unique(column_indices)
    if len(dynamic_rows) != len(dynamic_columns) or not len(dynamic_rows):
        raise RuntimeError('Descriptor reduction requires a square nonempty differential block')
    algebraic_rows = np.setdiff1d(np.arange(a.shape[0]), dynamic_rows)
    algebraic_columns = np.setdiff1d(np.arange(a.shape[1]), dynamic_columns)
    if len(algebraic_rows) != len(algebraic_columns):
        raise RuntimeError('Descriptor reduction requires a square algebraic block')
    gx = a[algebraic_rows][:, dynamic_columns]
    gz = a[algebraic_rows][:, algebraic_columns].tocsc()
    factor = splu(gz)
    sensitivity = -factor.solve(gx.toarray())
    reduced_a = (a[dynamic_rows][:, dynamic_columns].toarray()
                 + a[dynamic_rows][:, algebraic_columns] @ sensitivity)
    reduced_e = e[dynamic_rows][:, dynamic_columns].toarray()
    if np.linalg.matrix_rank(reduced_e) != len(dynamic_columns):
        raise RuntimeError('Reduced derivative matrix is singular; no index-one reduction is available')
    values, vectors = linalg.eig(reduced_a, reduced_e)
    if not np.all(np.isfinite(values)):
        raise RuntimeError('Reduced pencil returned nonfinite eigenvalues')
    full_vectors = np.empty((a.shape[1], len(values)), dtype=complex)
    full_vectors[dynamic_columns] = vectors
    full_vectors[algebraic_columns] = sensitivity @ vectors
    residuals = a @ full_vectors - (e @ full_vectors)*values[None, :]
    scales = (norm(a, np.inf) + abs(values)*norm(e, np.inf))*np.linalg.norm(full_vectors, axis=0)
    errors = np.linalg.norm(residuals, axis=0)/np.maximum(scales, 1e-30)
    worst = float(np.max(errors))
    if not np.isfinite(worst) or worst > 1e-8:
        raise RuntimeError(f'Full descriptor eigenpair residual exceeds tolerance: {worst}')
    return values, full_vectors, dict(
        descriptor_dimension=a.shape[0], reduced_dimension=len(dynamic_columns),
        algebraic_variables_eliminated=len(algebraic_columns),
        eigenpair_backward_error_max=worst, eigenpairs_validated=True,
        descriptor_regularization=0.,
        method='Sparse algebraic elimination + dense generalized eig; all finite modes')
