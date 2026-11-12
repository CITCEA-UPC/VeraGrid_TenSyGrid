"""Analytic matrix/mode derivatives checked against independent perturbations."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.linalg import eigvals
from hackaton_files import BasicRmsTuner, TunableGrid, TunableParameter
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Utils.Symbolic.symbolic import Const
from hackaton_files.test_rms_diagnostics import unstable
from hackaton_files.rms_sensitivity import build_parameterized_state_matrix, evaluate_state_matrix, state_matrix_derivative, eigenvalue_sensitivities, equilibrium_residual

def synthetic(*, shifting=False, repeated=False):
    vf = VarFactory()
    x, y, p = [vf.add_var(n) for n in ('x', 'y', 'p')]
    handle = TunableParameter('controller', 'device', 'p', 2.0, 0.1, 4.0)
    problem = SimpleNamespace(state_vars=[x], algebraic_vars=[y], _uid2idx_vars={x.uid: 0, y.uid: 1}, _variable_parameters=[p], _variable_parameters_values=np.array([2.0]), _constant_parameters=[], _constant_params=np.array([]), _event_parameter_device_idtags={p.uid: 'device'}, _state_eqs=[-p * x + y + (p - 2 if shifting else Const(0))], _algebraic_eqs=[x - p * y], get_x0=lambda: np.zeros(2), get_diff_var_number=lambda: 0, get_small_signal_reference_indices=lambda: None)
    if repeated:
        problem.state_vars = [x, y]
        problem.algebraic_vars = []
        problem._state_eqs = [-p * x, -p * y]
        problem._algebraic_eqs = []
    return (TunableGrid(problem, (handle,), (), (), (), ()), handle)

def test_symbolic_reduction_and_exact_derivative():
    grid, p = synthetic()
    tuner = BasicRmsTuner()
    model = tuner.get_parameterized_state_matrix(grid)
    for value in (1.0, 2.0, 3.0):
        np.testing.assert_allclose(evaluate_state_matrix(model, {p: value}), [[-value + 1 / value]])
        np.testing.assert_allclose(state_matrix_derivative(model, p, {p: value}), [[-1 - 1 / value ** 2]])
        sensitivity = eigenvalue_sensitivities(model, {p: value})
        assert sensitivity['valid_modes'][0]
        assert sensitivity['derivatives'][0, 0] == pytest.approx(-1 - 1 / value ** 2)
    assert model['parameter_symbols'][p].name == 'p'
    assert model['symbolic_blocks']['gy'][0, 0].get_vars()
    assert grid.problem._variable_parameters_values[0] == 2
    assert not evaluate_state_matrix(model).flags.writeable
    np.testing.assert_allclose(tuner.get_state_matrix(grid), [[-1.5]])
    np.testing.assert_allclose(tuner.get_state_matrix_derivative(grid, p), [[-1.25]])
    assert tuner.get_equilibrium_residual(grid) == 0
    assert tuner.get_eigenvalue_sensitivities(grid)['derivatives'][0, 0] == pytest.approx(-1.25)

def test_equilibrium_guard_bounds_and_snapshot():
    grid, p = synthetic(shifting=True)
    model = build_parameterized_state_matrix(grid)
    assert equilibrium_residual(model) == 0
    assert equilibrium_residual(model, {p: 3}) == pytest.approx(1)
    with pytest.raises(ValueError, match='not an equilibrium'):
        evaluate_state_matrix(model, {p: 3})
    np.testing.assert_allclose(evaluate_state_matrix(model, {p: 3}, require_equilibrium=False), [[-3 + 1 / 3]])
    for value in (np.nan, np.inf, 5, True):
        with pytest.raises(ValueError):
            evaluate_state_matrix(model, {p: value})
    with pytest.raises(ValueError, match='Unknown'):
        state_matrix_derivative(model, 'p')
    grid.problem._variable_parameters_values[0] = 3
    assert model['default_values'][p] == 2

def test_repeated_modes_are_not_assigned_individual_sensitivities():
    grid, _ = synthetic(repeated=True)
    result = eigenvalue_sensitivities(build_parameterized_state_matrix(grid))
    assert not result['valid_modes'].any()
    assert np.isnan(result['derivatives']).all()

def test_descriptor_rejection_is_explicit():
    grid, _ = synthetic()
    grid.problem.get_diff_var_number = lambda: 1
    with pytest.raises(NotImplementedError, match='descriptor'):
        build_parameterized_state_matrix(grid)

def test_real_grid_matrix_and_mode_derivatives(unstable):
    tuner, grid, baseline = unstable
    before = grid.problem._variable_parameters_values.copy()
    model = tuner.get_parameterized_state_matrix(grid)
    native_residual = np.r_[grid.problem.rhs_state(model['operating_point'], np.zeros_like(model['operating_point'])), grid.problem.rhs_algebraic(model['operating_point'], np.zeros_like(model['operating_point']))]
    assert equilibrium_residual(model) == pytest.approx(np.max(abs(native_residual)))
    np.testing.assert_allclose(evaluate_state_matrix(model, require_equilibrium=False), baseline['state_matrix'], atol=1e-09, rtol=1e-08)
    result = eigenvalue_sensitivities(model, require_equilibrium=False)
    worst = int(np.argmax(result['eigenvalues'].real))
    assert result['valid_modes'][worst]
    checked = 0
    for j, p in enumerate(model['parameters']):
        value = model['default_values'][p]
        h = 0.0001
        if not p.minimum < value - h < value + h < p.maximum:
            continue
        checked += 1
        plus = evaluate_state_matrix(model, {p: value + h}, require_equilibrium=False)
        from hackaton_files.rms_tuning import _parameter_indices
        from scipy.sparse.linalg import splu
        params = before.copy()
        params[_parameter_indices(grid.problem)[p.runtime_key]] = value + h
        args = (model['operating_point'], np.zeros_like(model['operating_point']), params, grid.problem._constant_params, 0.0)
        fx, fy, gx, gy = [fn(*args) for fn in (grid.problem._fx_ct_fn, grid.problem._fy_ct_fn, grid.problem._gx_ct_fn, grid.problem._gy_ct_fn)]
        native = fx.toarray() - fy @ splu(gy.tocsc()).solve(gx.toarray())
        np.testing.assert_allclose(plus, native, atol=1e-09, rtol=1e-08)
        minus = evaluate_state_matrix(model, {p: value - h}, require_equilibrium=False)
        np.testing.assert_allclose(state_matrix_derivative(model, p, require_equilibrium=False), (plus - minus) / (2 * h), atol=1e-07, rtol=1e-05)
        ep, em = (eigvals(plus), eigvals(minus))
        target = result['eigenvalues'][worst]
        numerical = (ep[np.argmin(abs(ep - target))] - em[np.argmin(abs(em - target))]) / (2 * h)
        np.testing.assert_allclose(result['derivatives'][worst, j], numerical, atol=1e-07, rtol=0.0001)
    assert checked > 0
    np.testing.assert_array_equal(grid.problem._variable_parameters_values, before)
    assert tuner.tests_used == grid.tests_used == 0


def test_grid_helpers_accept_overrides_and_refresh_runtime_values():
    grid, p = synthetic()
    tuner = BasicRmsTuner()
    np.testing.assert_allclose(tuner.get_state_matrix(grid, {p: 3.}), [[-3+1/3]])
    np.testing.assert_allclose(tuner.get_state_matrix_derivative(grid, p, {p: 3.}), [[-1-1/9]])
    result = tuner.get_eigenvalue_sensitivities(grid, values={p: 3.})
    assert result['derivatives'][0, 0] == pytest.approx(-1-1/9)
    grid.problem._variable_parameters_values[0] = 3.
    np.testing.assert_allclose(tuner.get_state_matrix(grid), [[-3+1/3]])
    assert tuner.tests_used == grid.tests_used == 0


def test_grid_helpers_preserve_equilibrium_guard():
    grid, p = synthetic(shifting=True)
    tuner = BasicRmsTuner()
    for call in (lambda: tuner.get_state_matrix(grid, {p: 3.}),
                 lambda: tuner.get_state_matrix_derivative(grid, p, {p: 3.}),
                 lambda: tuner.get_eigenvalue_sensitivities(grid, {p: 3.})):
        with pytest.raises(ValueError, match='not an equilibrium'):
            call()
    assert tuner.get_equilibrium_residual(grid, {p: 3.}) == pytest.approx(1.)
    result = tuner.get_eigenvalue_sensitivities(grid, {p: 3.}, require_equilibrium=False)
    assert result['equilibrium_residual'] == pytest.approx(1.)
