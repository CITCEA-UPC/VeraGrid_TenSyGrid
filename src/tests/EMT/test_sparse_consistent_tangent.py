"""Sparse DAE reduction must preserve the dense minimum-norm tangent."""

import numpy as np
import pytest
from scipy import linalg, sparse
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_consistent_tangent import _sparse_algebraic_tangent


@pytest.mark.parametrize('permuted', [False, True])
def test_sparse_tangent_matches_dense_constraint_space_and_minimum_norm(permuted: bool) -> None:
    gz = np.array([[1., 2., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0.],
                   [0., 0., 2., 0., 0., 0.],
                   [0., 0., 0., 3., 0., 0.],
                   [0., 0., 0., 0., 4., 0.]])
    gx = np.random.default_rng(42).normal(size=(6, 4))
    if permuted:
        rows, columns = [3, 0, 5, 1, 4, 2], [4, 1, 3, 0, 5, 2]
        gz = gz[rows][:, columns]
        gx = gx[rows]
    result = _sparse_algebraic_tangent(sparse.csc_matrix(gx), sparse.csc_matrix(gz))
    assert result is not None
    basis, algebraic = result
    left_null = linalg.null_space(gz.T)
    expected_basis = linalg.null_space(left_null.T @ gx)
    np.testing.assert_allclose(basis @ basis.T, expected_basis @ expected_basis.T, atol=2e-12)
    np.testing.assert_allclose(algebraic, -linalg.pinv(gz) @ gx @ basis, atol=2e-12)
    np.testing.assert_allclose(gx @ basis + gz @ algebraic, 0., atol=2e-12)


def test_sparse_tangent_handles_invertible_algebraic_block() -> None:
    gz = np.array([[2., 1.], [1., 3.]])
    gx = np.array([[1., 2., 3.], [4., 5., 6.]])
    result = _sparse_algebraic_tangent(sparse.csc_matrix(gx), sparse.csc_matrix(gz))
    assert result is not None
    basis, algebraic = result
    np.testing.assert_allclose(basis, np.eye(3))
    np.testing.assert_allclose(algebraic, -linalg.solve(gz, gx))


def test_sparse_tangent_defers_numerically_singular_minor_to_dense_path() -> None:
    result = _sparse_algebraic_tangent(sparse.eye(2, format='csc'),
                                      sparse.csc_matrix([[1., 1.], [2., 2.]]))
    assert result is None
