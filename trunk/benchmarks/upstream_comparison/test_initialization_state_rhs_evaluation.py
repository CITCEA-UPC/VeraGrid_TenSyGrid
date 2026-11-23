"""Check numerical equivalence of initialization-only RHS evaluation backends."""

import numpy as np
import pytest
from collections.abc import Callable
from VeraGridEngine.Utils.Symbolic.jit_compiler import RuntimeParameterCompiler
from VeraGridEngine.Utils.Symbolic.symbolic import Var, Expr
from VeraGridEngine.enumerations import DynamicIntegrationMethod
from VeraGridEngine.Simulations.EMT.solvers.StructuralVectorizedSolver import SparseADJacobian
from VeraGridEngine.Simulations.EMT.initialization_emt import InitializationStateRhsVector


@pytest.mark.parametrize('size', [1, 32])
def test_initial_rhs_python_matches_jit_for_updated_inputs(size: int) -> None:
    """Compare both evaluators across repeated state and parameter updates.

    :param size: Number of state equations, including a network-sized threshold case.
    :return: None.
    """
    lines: list[str] = ['def func(values, diffs, events, params, out):']
    for index in range(size):
        lines.append(
            f'    out[{index}] = (values[{index}] - diffs[{index}]) * '
            f'(params[{index}] + events[{index}]) + math.sin(values[{index}]) '
            f'+ math.sqrt(1.0 + values[{index}] * values[{index}])')
    source: str = '\n'.join(lines) + '\n'
    interpreted: InitializationStateRhsVector = InitializationStateRhsVector(source, False, size)
    compiled: InitializationStateRhsVector = InitializationStateRhsVector(source, True, size)
    values: np.ndarray = np.linspace(-2.0, 2.0, size)
    diffs: np.ndarray = np.linspace(0.1, 0.5, size)
    events: np.ndarray = np.linspace(-0.4, 0.7, size)
    params: np.ndarray = np.linspace(0.2, 1.1, size)
    iteration: int
    for iteration in range(3):
        actual: np.ndarray = interpreted(values, diffs, events, params).copy()
        expected: np.ndarray = compiled(values, diffs, events, params).copy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        values += 0.15
        events -= 0.07
        params += 0.03


def test_runtime_python_option_preserves_sequential_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare runtime backends when later equations consume earlier writes.

    :param monkeypatch: Scoped environment configuration for the compiler option.
    :return: None.
    """
    first: Var = Var('first')
    second: Var = Var('second')
    fixed: Var = Var('fixed')
    tm: Var = Var('time')
    compiler: RuntimeParameterCompiler = RuntimeParameterCompiler(
        runtime_map={first.uid: 0, second.uid: 1}, fixed_map={fixed.uid: 0},
        time_uids={tm.uid})
    equations: list[Expr] = [first + fixed + tm, first * second]
    with monkeypatch.context() as configuration:
        configuration.setenv('VERAGRID_RUNTIME_PARAMETER_PYTHON', '1')
        interpreted: Callable = compiler.compile(equations)
    with monkeypatch.context() as configuration:
        configuration.delenv('VERAGRID_RUNTIME_PARAMETER_PYTHON', raising=False)
        compiled: Callable = compiler.compile(equations)
    actual: np.ndarray = np.array([1.0, 2.0])
    expected: np.ndarray = actual.copy()
    fixed_values: np.ndarray = np.array([3.0])
    iteration: int
    for iteration in range(3):
        interpreted(actual, fixed_values, 0.5)
        compiled(expected, fixed_values, 0.5)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, [11.5, 828.0])


def test_structural_python_jacobian_matches_compiled_and_analytic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check the opt-in kernel path against a nonlinear implicit DAE Jacobian.

    :param monkeypatch: Scoped backend configuration for the two evaluators.
    :return: None.
    """
    method = DynamicIntegrationMethod.DaeBackEuler
    first: Var = Var('first_state')
    second: Var = Var('second_state')
    derivative: Var = Var('derivative', base_var=first)
    parameter: Var = Var('parameter')
    equations: list[Expr] = [derivative + first * second + parameter, second * second - first]
    with monkeypatch.context() as configuration:
        configuration.delenv('VERAGRID_STRUCTURAL_PYTHON', raising=False)
        compiled = SparseADJacobian(equations, [first, second], [parameter], method, verbose=False)
    with monkeypatch.context() as configuration:
        configuration.setenv('VERAGRID_STRUCTURAL_PYTHON', '1')
        interpreted = SparseADJacobian(equations, [first, second], [parameter], method, verbose=False)
    states: np.ndarray = np.array([1.5, 2.0])
    history: np.ndarray = np.array([1.4, 1.9])
    d_history: np.ndarray = np.array([0.1, 0.2])
    parameters: np.ndarray = np.array([0.3])
    expected: np.ndarray = np.array([[102.0, 1.5], [-1.0, 4.0]])
    actual: np.ndarray = interpreted(states, parameters, history, d_history, 0.01).toarray()
    reference: np.ndarray = compiled(states, parameters, history, d_history, 0.01).toarray()
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
