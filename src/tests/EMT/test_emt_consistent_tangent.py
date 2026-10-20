"""Small DAE checks for consistent Floquet initial perturbations."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_consistent_tangent import (
    ProjectedEmtFloquetOperator,
    build_consistent_emt_tangent,
)
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_floquet_operator import BlockEmtFloquetOperator
from VeraGridEngine.enumerations import DynamicIntegrationMethod


@dataclass(frozen=True)
class _Var:
    uid: int
    base_var: object = None


class _ConstrainedProblem:
    """xdot=-2x, ydot=z, y=0: only x is admissible."""

    def __init__(self):
        self.states = [_Var(1), _Var(2)]
        self.diffs = [_Var(3, self.states[0]), _Var(4, self.states[1])]

    def get_state_vars(self):
        return self.states

    def get_diff_vars(self):
        return self.diffs

    def get_states_number(self):
        return 2

    def get_algebraic_var_number(self):
        return 1


def test_consistent_tangent_restricts_hidden_constraint_and_seeds_derivative():
    problem = _ConstrainedProblem()
    h = 0.1

    def jacobian(*, h, **_kwargs):
        return sparse.csc_matrix([
            [2 / h + 2, 0, 0],
            [0, 2 / h, -1],
            [0, 1, 0],
        ])

    trajectory = np.zeros((2, 3))
    derivative = np.zeros_like(trajectory)
    tangent = build_consistent_emt_tangent(
        problem, trajectory, derivative, np.array([0.0, h]), h,
        jacobian, np.empty(0), 0,
    )
    assert tangent.basis.shape == (2, 1)
    np.testing.assert_allclose(tangent.basis[1], 0, atol=1e-14)
    np.testing.assert_allclose(
        tangent.initial_derivative_map @ tangent.basis,
        -2 * tangent.basis,
        atol=1e-14,
    )

    full = BlockEmtFloquetOperator(
        problem=problem, trajectory=trajectory, h=h, n_states=2,
        method=DynamicIntegrationMethod.DaeTrapezoidal,
        jac_evaluator=jacobian, static_params=np.empty(0), n_event_params=0,
        t_trajectory=np.array([0.0, h]), d_trajectory=derivative,
        initial_tangent_derivative=tangent.initial_derivative_map,
    )
    projected = ProjectedEmtFloquetOperator(full, tangent.basis)
    expected = (1 - h) / (1 + h)
    np.testing.assert_allclose(projected.matvec(np.ones(1)), [expected], atol=1e-14)
