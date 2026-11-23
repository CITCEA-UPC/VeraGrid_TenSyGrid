"""Check the sparse Floquet adapter against a discrete residual derivative."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from VeraGridEngine.Simulations.EMT.problems.emt_problem_multilinear import EmtProblemMultilinear
from VeraGridEngine.Simulations.Rms.problems.rms_problem_multilinear import RmsProblemMultilinear
from VeraGridEngine.enumerations import DynamicIntegrationMethod


class PolynomialProblem:
    """xdot=-2*x+x*z, 0=z-3*x, with explicit multilinear monomials."""
    get_floquet_jacobian_evaluator = EmtProblemMultilinear.get_floquet_jacobian_evaluator
    _compute_jacobian_sparse_from_S = staticmethod(RmsProblemMultilinear._compute_jacobian_sparse_from_S)

    def __init__(self, method):
        x, z = SimpleNamespace(uid=1), SimpleNamespace(uid=2)
        dx = SimpleNamespace(uid=3, base_var=x)
        self._state_vars, self._algebraic_vars, self._diff_vars = [x], [z], [dx]
        self._ml_all_vars_sa = [x, z]
        self.options = SimpleNamespace(integration_method=method, time_step=.01, explicit=False)

    def _ensure_multilinear_index_cache(self):
        pass

    def build_multilinear_matrices(self):
        return (sparse.csc_matrix([[1., 0., 1.], [0., 1., 1.], [0., 0., 0.]]),
                sparse.csc_matrix([[-2., 0., 1.], [-3., 1., 0.]]))


@pytest.mark.parametrize('method,coefficient', [
    (DynamicIntegrationMethod.DaeBackEuler, 1.),
    (DynamicIntegrationMethod.DaeTrapezoidal, 2.),
    (DynamicIntegrationMethod.DaeBDF2, 1.5),
])
@pytest.mark.parametrize('point', [[.8, .7], [0., 0.]])
def test_sparse_adapter_matches_independent_discrete_residual(method, coefficient, point):
    problem = PolynomialProblem(method)
    step, epsilon = .03, 1e-6
    alpha = coefficient / step
    def residual(values):
        x, z = values
        return np.array([alpha * x - (-2*x + x*z), z - 3*x])
    point = np.array(point)
    expected = np.column_stack([(residual(point + epsilon * direction)
                                - residual(point - epsilon * direction))/(2 * epsilon)
                               for direction in np.eye(2)])
    actual = problem.get_floquet_jacobian_evaluator()(states=point, h=step)
    assert sparse.issparse(actual)
    np.testing.assert_allclose(actual.toarray(), expected, atol=1e-8)
