"""Check compiled monomial derivatives against independent product differentiation."""
import numpy as np
import pytest
from scipy import sparse
from VeraGridEngine.Simulations.Rms.problems.rms_problem_multilinear import (
    RmsProblemMultilinear, _multilinear_jacobian_csc_numba,
)


@pytest.mark.parametrize('zero_fraction', [0., .2, 1.])
@pytest.mark.parametrize('index_dtype', [np.int32, np.int64])
def test_compiled_monomial_derivatives_preserve_zero_factors(zero_fraction, index_dtype):
    rng = np.random.default_rng(8)
    coefficients = rng.choice([-1., 0., 0., 0., 1.], size=(20, 30))
    coefficients[:, 0] = 0.  # A constant monomial has no derivatives.
    matrix = sparse.csc_matrix(coefficients)
    matrix.indices = matrix.indices.astype(index_dtype)
    matrix.indptr = matrix.indptr.astype(index_dtype)
    values = rng.normal(size=20)
    values[rng.random(20) < zero_fraction] = 0.
    expected = np.zeros((20, 30))
    for column in range(30):
        active = np.flatnonzero(coefficients[:, column])
        for row in active:
            # Differentiate a product by multiplying all other factors, without division.
            product = coefficients[row, column]
            for other in active:
                if other != row:
                    coefficient = coefficients[other, column]
                    product *= coefficient * values[other] + 1 - abs(coefficient)
            expected[row, column] = product
    actual = RmsProblemMultilinear._compute_jacobian_sparse_from_S(matrix, values)
    np.testing.assert_allclose(actual.toarray(), expected, atol=1e-14)
    assert _multilinear_jacobian_csc_numba.nopython_signatures


@pytest.mark.parametrize('values,expected', [
    ([5e-13, 2.], [2., 5e-13]),
    ([5e-13, 3e-13], [0., 0.]),
])
def test_compiled_kernel_keeps_existing_near_zero_factor_rule(values, expected):
    matrix = sparse.csc_matrix([[1.], [1.]])
    actual = RmsProblemMultilinear._compute_jacobian_sparse_from_S(matrix, np.array(values))
    np.testing.assert_array_equal(actual.toarray().ravel(), np.array(expected))
