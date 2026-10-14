"""Fit CP factors to genuine degree-five IEEE-9 converter interactions.

The five input modes are *groups of physical states*, each contracted with a
linear form. This is CP for the grouped multilinear block. It is not a CP
decomposition of the full 49-binary-mode coefficient tensor.
"""

from __future__ import annotations

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

from TensyGridEngine.emt.emt_explicit_converter_demo.tensor_decompositions.ieee9_gfl_ode_degree5_tt import (
    CONVERTERS, NETWORK, degree_five_block,
)
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_gfl_ode_reduced import build_problem


def evaluate_group_cp(weights, factors, grouped_values):
    """Evaluate eight products of five linear forms and six output weights."""
    products = weights.copy()
    for factor, values in zip(factors[1:], grouped_values):
        products *= values @ factor
    return factors[0] @ products


def fit_rank_eight(block, tolerance=1e-11):
    """Restart CP-ALS to avoid treating one local minimum as a rank result."""
    best = None
    for initialization in ("svd", 0, 1, 2, 3, 4):
        kwargs = ({"init": "svd", "random_state": 1} if initialization == "svd"
                  else {"init": "random", "random_state": initialization})
        decomposition = parafac(
            block, rank=8, n_iter_max=2000, tol=1e-14,
            normalize_factors=True, **kwargs,
        )
        error = np.linalg.norm(tl.cp_to_tensor(decomposition) - block) / np.linalg.norm(block)
        if best is None or error < best[0]:
            best = (error, decomposition, initialization)
        if error < tolerance:
            return best
    return best


def main() -> None:
    reduced = build_problem().build_reduced_model_problem()
    S, phi = reduced.build_multilinear_matrices_fast()
    n_state = len(reduced.get_state_vars())
    monomials = [
        tuple(int(index) for index in S.indices[S.indptr[column]:S.indptr[column + 1]]
              if index < n_state)
        for column in range(S.shape[1])
    ]
    rng = np.random.default_rng(20261009)
    for converter, (rows, groups) in enumerate(CONVERTERS, start=1):
        block, supports = degree_five_block(phi, monomials, rows, groups)
        error, decomposition, initialization = fit_rank_eight(block)
        weights, factors = decomposition
        evaluation_errors = []
        for _ in range(20):
            states = rng.standard_normal(n_state) * 0.1
            values = [states[list(group)] for group in (NETWORK, *groups)]
            direct = np.einsum("eabcdf,a,b,c,d,f->e", block, *values)
            cp_value = evaluate_group_cp(weights, factors, values)
            evaluation_errors.append(
                np.linalg.norm(cp_value - direct) / max(np.linalg.norm(direct), 1e-14)
            )
        if error > 1e-11 or max(evaluation_errors) > 1e-10:
            raise RuntimeError(f"Converter {converter} rank-eight CP fit was not accurate")
        stored = weights.size + sum(factor.size for factor in factors)
        print(f"converter {converter}: degree-five supports={supports}, "
              f"grouped shape={block.shape}")
        print(f"  grouped CP terms=8, stored floats={stored}, "
              f"initialization={initialization}")
        print(f"  coefficient error={error:.3e}, "
              f"max evaluation error={max(evaluation_errors):.3e}")


if __name__ == "__main__":
    main()
