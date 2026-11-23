"""Symbolic parameter dependence and fixed-operating-point RMS sensitivities."""
from __future__ import annotations
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping
import numbers
import numpy as np
from scipy.linalg import eig
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from .rms_diagnostics import _readonly
if TYPE_CHECKING:
    from .rms_tuning import TunableGrid, TunableParameter

def build_parameterized_state_matrix(grid: TunableGrid) -> dict:
    """Snapshot symbolic fx/fy/gx/gy, editable symbols and fixed-point bindings.

    Supports semi-explicit DAEs. Underscore keys are evaluation/cache details.
    """
    model = {}
    from .rms_tuning import TunableGrid, _parameter_indices
    if not isinstance(grid, TunableGrid):
        raise TypeError('Expected a TunableGrid with an editable parameter catalog')
    problem = grid.problem
    if problem.get_diff_var_number():
        raise NotImplementedError('Parameterized reduction currently requires a semi-explicit DAE; descriptor sensitivities need both A(p) and E(p)')
    model['parameters'] = tuple(grid.tunable_parameters)
    indices = _parameter_indices(problem)
    if any((p.runtime_key not in indices for p in model['parameters'])):
        raise ValueError('Editable parameter is missing from the initialized problem')
    model['parameter_symbols'] = MappingProxyType({p: problem._variable_parameters[indices[p.runtime_key]] for p in model['parameters']})
    model['default_values'] = MappingProxyType({p: float(problem._variable_parameters_values[indices[p.runtime_key]]) for p in model['parameters']})
    model['state_variables'] = tuple(problem.state_vars)
    model['algebraic_variables'] = tuple(problem.algebraic_vars)
    model['operating_point'] = _readonly(problem.get_x0())
    ns = len(model['state_variables'])
    model['_reference'] = problem.get_small_signal_reference_indices()
    if model['_reference'] is not None:
        model['_reference'] = tuple((i - ns for i in model['_reference']))
    editable_uids = {v.uid for v in model['parameter_symbols'].values()}
    substitutions = {v: Const(float(model['operating_point'][problem._uid2idx_vars[v.uid]])) for v in model['state_variables'] + model['algebraic_variables']}
    for variables, values in ((problem._variable_parameters, problem._variable_parameters_values), (problem._constant_parameters, problem._constant_params)):
        substitutions.update({v: Const(float(value)) for v, value in zip(variables, values) if v.uid not in editable_uids})
    f, g = (list(problem._state_eqs), list(problem._algebraic_eqs))

    def symbolic_jacobian(equations, variables):
        result = np.empty((len(equations), len(variables)), dtype=object)
        for i, equation in enumerate(equations):
            active = {v.uid for v in equation.get_vars()}
            for j, variable in enumerate(variables):
                result[i, j] = equation.diff(variable).subs(substitutions).simplify() if variable.uid in active else Const(0)
        result.flags.writeable = False
        return result
    model['symbolic_blocks'] = MappingProxyType({'fx': symbolic_jacobian(f, model['state_variables']), 'fy': symbolic_jacobian(f, model['algebraic_variables']), 'gx': symbolic_jacobian(g, model['state_variables']), 'gy': symbolic_jacobian(g, model['algebraic_variables'])})
    model['_residual'] = tuple((eq.subs(substitutions).simplify() for eq in f + g))
    for expression in [*model['_residual'], *(e for block in model['symbolic_blocks'].values() for e in block.flat)]:
        if any((v.uid not in editable_uids for v in expression.get_vars())):
            raise ValueError('Unbound symbol remains after freezing the operating point')
    model['_derivatives'] = {}
    return model

def _bindings(model, values):
    proposed = dict(model['default_values'])
    for parameter, value in ({} if values is None else values).items():
        if parameter not in proposed:
            raise ValueError('Unknown editable parameter handle')
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or (not np.isfinite(value)):
            raise ValueError('Parameter values must be finite real numbers')
        if not parameter.minimum <= value <= parameter.maximum:
            raise ValueError(f'{parameter.key} is outside its tuning range')
        proposed[parameter] = float(value)
    return {model['parameter_symbols'][p].uid: value for p, value in proposed.items()}

def _evaluate_blocks(blocks, bindings):
    result = {name: np.array([e.eval_uid(bindings) for e in block.flat], dtype=float).reshape(block.shape) for name, block in blocks.items()}
    if any((not np.isfinite(block).all() for block in result.values())):
        raise ValueError('Nonfinite parameterized Jacobian')
    return result

def equilibrium_residual(model, values: Mapping[TunableParameter, float] | None=None) -> float:
    """Infinity norm of f,g at the frozen point under proposed parameters."""
    bindings = _bindings(model, values)
    residual = np.array([eq.eval_uid(bindings) for eq in model['_residual']], dtype=float)
    if not np.isfinite(residual).all():
        return float('inf')
    return float(np.max(np.abs(residual), initial=0))

def _prepare(model, values, require_equilibrium, equilibrium_tolerance):
    if not np.isfinite(equilibrium_tolerance) or equilibrium_tolerance < 0:
        raise ValueError('equilibrium_tolerance must be finite and nonnegative')
    bindings = _bindings(model, values)
    if require_equilibrium and equilibrium_residual(model, values) > equilibrium_tolerance:
        raise ValueError('The frozen operating point is not an equilibrium for these parameters; reinitialize the grid or explicitly set require_equilibrium=False for a local Jacobian')
    blocks = _evaluate_blocks(model['symbolic_blocks'], bindings)
    constrained_row = None
    if blocks['gy'].shape[0] == 0:
        return (bindings, blocks, None, blocks['gx'].copy(), constrained_row)
    try:
        solver = splu(csc_matrix(blocks['gy']))
    except RuntimeError:
        if model['_reference'] is None:
            raise ValueError('Algebraic Jacobian is singular; state reduction is unavailable') from None
        row, col = model['_reference']
        if not (0 <= row < blocks['gy'].shape[0] and 0 <= col < blocks['gy'].shape[1]):
            raise ValueError('Invalid algebraic reference indices')
        blocks['gy'][row, :] = 0
        blocks['gy'][row, col] = 1
        blocks['gx'][row, :] = 0
        try:
            solver = splu(csc_matrix(blocks['gy']))
        except RuntimeError as exc:
            raise ValueError('Algebraic Jacobian remains singular after fixing the angle reference') from exc
        constrained_row = row
    coupling = solver.solve(blocks['gx'])
    return (bindings, blocks, solver, coupling, constrained_row)

def evaluate_state_matrix(model, values=None, *, require_equilibrium=True, equilibrium_tolerance=1e-06):
    """Evaluate A(p)=fx-fy solve(gy,gx); unspecified controls retain defaults."""
    _, b, _, coupling, _ = _prepare(model, values, require_equilibrium, equilibrium_tolerance)
    return _readonly(b['fx'] - b['fy'] @ coupling)

def _derivative(model, parameter, prepared):
    if parameter not in model['parameter_symbols']:
        raise ValueError('Unknown editable parameter handle')
    if parameter not in model['_derivatives']:
        symbol = model['parameter_symbols'][parameter]
        model['_derivatives'][parameter] = {name: np.array([e.diff(symbol).simplify() for e in block.flat], dtype=object).reshape(block.shape) for name, block in model['symbolic_blocks'].items()}
    bindings, b, solver, coupling, row = prepared
    d = _evaluate_blocks(model['_derivatives'][parameter], bindings)
    if row is not None:
        d['gy'][row, :] = 0
        d['gx'][row, :] = 0
    # Differentiate gy @ coupling = gx, using the same factorization.
    dcoupling = d['gx'] if solver is None else solver.solve(d['gx'] - d['gy'] @ coupling)
    return d['fx'] - d['fy'] @ coupling - b['fy'] @ dcoupling

def state_matrix_derivative(model, parameter, values=None, *, require_equilibrium=True, equilibrium_tolerance=1e-06):
    """Analytic partial dA/dp at the frozen operating point (no finite differences)."""
    prepared = _prepare(model, values, require_equilibrium, equilibrium_tolerance)
    return _readonly(_derivative(model, parameter, prepared))

def eigenvalue_sensitivities(model, values=None, *, require_equilibrium=True, equilibrium_tolerance=1e-06, separation_tolerance=1e-07):
    """For simple modes compute dλ/dp = lᴴ(dA/dp)r / (lᴴr).

        Repeated/near-repeated and ill-conditioned eigenpairs are marked invalid;
        no reliable individual-mode derivative is assigned to these modes.
        """
    if not np.isfinite(separation_tolerance) or separation_tolerance <= 0:
        raise ValueError('separation_tolerance must be finite and positive')
    prepared = _prepare(model, values, require_equilibrium, equilibrium_tolerance)
    _, b, _, coupling, _ = prepared
    A = b['fx'] - b['fy'] @ coupling
    # Left/right eigenvectors must be in the original state coordinates.
    eigenvalues, left, right = eig(A, left=True, right=True)
    valid = np.ones(len(eigenvalues), dtype=bool)
    denominators = np.sum(left.conj() * right, axis=0)
    for i, value in enumerate(eigenvalues):
        others = np.delete(eigenvalues, i)
        if np.any(np.abs(value - others) <= separation_tolerance * np.maximum(1, np.maximum(abs(value), abs(others)))) or abs(denominators[i]) <= 1e-08:
            valid[i] = False
    derivatives = np.full((len(eigenvalues), len(model['parameters'])), np.nan + 1j * np.nan)
    for j, parameter in enumerate(model['parameters']):
        dA = _derivative(model, parameter, prepared)
        for i in np.flatnonzero(valid):
            derivatives[i, j] = np.vdot(left[:, i], dA @ right[:, i]) / denominators[i]
    return dict(eigenvalues=_readonly(eigenvalues), parameters=model['parameters'], derivatives=_readonly(derivatives), valid_modes=_readonly(valid), equilibrium_residual=equilibrium_residual(model, values))
