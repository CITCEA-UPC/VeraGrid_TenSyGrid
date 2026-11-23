"""Check numerical equivalence of initialization-only RHS evaluation backends."""

import numpy as np
import pytest
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
