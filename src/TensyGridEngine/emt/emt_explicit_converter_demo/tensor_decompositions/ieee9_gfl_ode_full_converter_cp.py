"""Exact binary-mode CP baseline for the complete reduced IEEE-9 model.

Both 17-equation converters and the complete 49-equation model are checked.
All polynomial degrees are included. Each used monomial supplies one CP term:
its output factor is the vector of equation coefficients, and its 49 binary
factors select degree zero or one in each state. The conceptual tensor is
never allocated densely.

This constructs a valid full-expression CP representation and verifies it
against the fast S/Phi extraction. It does not claim minimal CP rank.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from TensyGridEngine.emt.emt_explicit_converter_demo.tensor_decompositions.ieee9_gfl_ode_cp_rank_probe import unfolding_rank
from TensyGridEngine.emt.emt_explicit_converter_demo.ieee9_gfl_ode_reduced import build_problem


CONVERTER_ROWS = (slice(6, 23), slice(23, 40))


def exact_monomial_cp(phi: sparse.csr_matrix, monomials: list[tuple[int, ...]]):
    """Return nonzero output factors and sparse binary-mode factor locations."""
    used_columns = np.unique(phi.indices)
    output_factors = phi[:, used_columns].toarray()
    supports = [monomials[int(column)] for column in used_columns]
    return output_factors, supports


def evaluate_cp(output_factors: np.ndarray, supports: list[tuple[int, ...]],
                states: np.ndarray) -> np.ndarray:
    """Contract factors [0,1] at selected modes and [1,0] elsewhere."""
    products = np.array([np.prod(states[list(support)]) if support else 1.0
                         for support in supports])
    return output_factors @ products


def evaluate_sparse_phi(phi: sparse.csr_matrix, monomials: list[tuple[int, ...]],
                        states: np.ndarray) -> np.ndarray:
    """Independent S/Phi monomial contraction over all extracted columns."""
    values = np.array([np.prod(states[list(monomial)]) if monomial else 1.0
                       for monomial in monomials])
    return phi @ values


def count_one_mode_merges(outputs: np.ndarray, supports: list[tuple[int, ...]],
                          tolerance: float = 1e-12) -> tuple[int, int]:
    """Count pairs that could share one affine binary-mode CP factor.

    A necessary condition for this elementary merge is collinearity of the
    output factors. More general CP factorizations may still compress them.
    """
    by_support = {frozenset(support): column
                  for column, support in enumerate(supports)}
    candidates = merges = 0
    for column, support in enumerate(supports):
        output = outputs[:, column]
        output_norm = np.linalg.norm(output)
        for mode in support:
            other = by_support.get(frozenset(support) - {mode})
            if other is None:
                continue
            other_output = outputs[:, other]
            other_norm = np.linalg.norm(other_output)
            candidates += 1
            difference = np.linalg.norm(
                output / output_norm
                - np.sign(output @ other_output) * other_output / other_norm
            )
            merges += difference < tolerance
    return candidates, merges


def main() -> None:
    reduced = build_problem().build_reduced_model_problem()
    S, full_phi = reduced.build_multilinear_matrices_fast()
    n_state = len(reduced.get_state_vars())
    monomials = [
        tuple(int(index) for index in S.indices[S.indptr[column]:S.indptr[column + 1]]
              if index < n_state)
        for column in range(S.shape[1])
    ]
    rng = np.random.default_rng(20261009)
    for label, rows in (("converter 1", CONVERTER_ROWS[0]),
                        ("converter 2", CONVERTER_ROWS[1]),
                        ("all 49 equations", slice(0, 49))):
        phi = full_phi[rows].tocsr()
        outputs, supports = exact_monomial_cp(phi, monomials)
        used = np.unique(phi.indices)

        # Rebuild every coefficient column from the CP output factors. The
        # binary input factors have exactly one nonzero tensor entry per term.
        reconstructed = sparse.csr_matrix(
            (outputs.ravel(),
             (np.repeat(np.arange(phi.shape[0]), len(used)), np.tile(used, phi.shape[0]))),
            shape=phi.shape,
        )
        coefficient_error = np.linalg.norm((reconstructed - phi).data)
        evaluation_errors = []
        for _ in range(20):
            states = rng.standard_normal(n_state) * 0.1
            cp_value = evaluate_cp(outputs, supports, states)
            reference = evaluate_sparse_phi(phi, monomials, states)
            evaluation_errors.append(np.linalg.norm(cp_value - reference)
                                     / max(np.linalg.norm(reference), 1e-14))
        if coefficient_error != 0 or max(evaluation_errors) > 1e-12:
            raise RuntimeError(f"{label} full CP verification failed")

        degrees = Counter(len(support) for support in supports)
        alternating_shape, alternating_rank = unfolding_rank(
            phi, monomials, set(range(0, n_state, 2))
        )
        candidates, merges = count_one_mode_merges(outputs, supports)
        print(f"{label}: outputs={phi.shape[0]}, state modes={n_state}, "
              f"CP terms={len(supports)}, coefficient entries={phi.nnz}")
        print(f"  supports by degree={dict(sorted(degrees.items()))}")
        print(f"  alternating unfolding={alternating_shape}, CP rank >= {alternating_rank}")
        print(f"  full-expression CP rank <= {len(supports)}")
        print(f"  elementary one-mode merge candidates={candidates}, "
              f"collinear pairs={merges}")
        print(f"  coefficient error={coefficient_error:.1e}, "
              f"max evaluation error={max(evaluation_errors):.3e}")
        print(f"  dense CP factor scalars={outputs.size + 2 * n_state * len(supports)}, "
              f"sparse output entries={phi.nnz}, selected input modes={sum(map(len, supports))}")


if __name__ == "__main__":
    main()
