"""Check the automatic compact AD backend against generated AD and differences."""

from unittest.mock import patch

import numpy as np
import pytest

from VeraGridEngine.Simulations.EMT.solvers.StructuralVectorizedSolver import SparseADJacobian
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Func, Var
from VeraGridEngine.enumerations import DynamicIntegrationMethod


@pytest.mark.parametrize('method', [DynamicIntegrationMethod.DaeTrapezoidal,
                                    DynamicIntegrationMethod.DaeBackEuler,
                                    DynamicIntegrationMethod.DaeBDF2])
def test_automatic_tape_matches_generated_ad_and_residual_differences(method):
    variables = [Var(f'x{i}') for i in range(24)]
    gain = Var('gain')
    equations = []
    differentials = []
    for i, x in enumerate(variables):
        y = variables[(i + 1) % len(variables)]
        dx = Var(f'dx{i}', base_var=x)
        differentials.append(dx)
        # Include nonlinear derivative coefficients, repeated factors, runtime
        # parameters and unary functions; these exercise residual history too.
        equations.append((1 + x * x) * dx - (
            gain * x * x * y + Func(y, 'sin') + x / (2 + y)
            + Func(x, 'exp') + Func(2 + x, 'sqrt')
            + Func(3 + y, 'log') + Func(x, 'abs')))

    # This branch stores derivative history in differential order, which may
    # differ from state order. Both residuals and Jacobians must retain it.
    differentials.reverse()
    tape = SparseADJacobian(equations, variables, [gain], differentials, method)
    assert tape._tape is not None
    with patch(
            'VeraGridEngine.Simulations.EMT.solvers.StructuralVectorizedSolver.build_structural_ad_tape',
            return_value=None), patch(
            'VeraGridEngine.Simulations.EMT.solvers.StructuralVectorizedSolver._safe_njit',
            side_effect=lambda function, **kwargs: function):
        generated = SparseADJacobian(equations, variables, [gain], differentials, method)

    x0 = np.linspace(-0.4, 0.7, len(variables))
    history = x0 * 0.9
    derivative_history = np.linspace(-0.2, 0.3, len(variables))
    history2 = x0 * 0.8
    h = 0.015
    for gain_value in (0.7, -1.4):
        params = np.array([gain_value])
        args = dict(states=x0, params=params, history=history,
                    d_history=derivative_history, h=h, history2=history2)
        actual = tape(**args).toarray()
        np.testing.assert_allclose(actual, generated(**args).toarray(), rtol=1e-12, atol=1e-12)

        numerical = np.empty_like(actual)
        epsilon = 1e-6
        for col in range(len(variables)):
            plus = x0.copy()
            minus = x0.copy()
            plus[col] += epsilon
            minus[col] -= epsilon
            f_plus = np.empty(len(variables))
            f_minus = np.empty(len(variables))
            tape._tape.residual(plus, params, history, derivative_history, h, f_plus, history2)
            tape._tape.residual(minus, params, history, derivative_history, h, f_minus, history2)
            numerical[:, col] = (f_plus - f_minus) / (2 * epsilon)
        np.testing.assert_allclose(actual, numerical, rtol=1e-8, atol=2e-8)


def test_dynamic_power_retains_generated_backend():
    variables = [Var(f'x{i}') for i in range(20)]
    equations = [x ** variables[(i + 1) % len(variables)] for i, x in enumerate(variables)]
    with patch(
            'VeraGridEngine.Simulations.EMT.solvers.StructuralVectorizedSolver._safe_njit',
            side_effect=lambda function, **kwargs: function):
        jacobian = SparseADJacobian(equations, variables, [], [],
                                    DynamicIntegrationMethod.DaeTrapezoidal)
    assert jacobian._tape is None


def test_polynomial_at_zero_keeps_repeated_power_derivatives():
    variables = [Var(f'x{i}') for i in range(20)]
    equations = [x * x + x ** Const(3) for x in variables]
    jacobian = SparseADJacobian(equations, variables, [], [],
                                DynamicIntegrationMethod.DaeTrapezoidal)
    assert jacobian._tape is not None
    x0 = np.zeros(len(variables))
    actual = jacobian(x0, np.empty(0), x0, x0, 0.01).toarray()
    np.testing.assert_array_equal(actual, np.zeros_like(actual))
