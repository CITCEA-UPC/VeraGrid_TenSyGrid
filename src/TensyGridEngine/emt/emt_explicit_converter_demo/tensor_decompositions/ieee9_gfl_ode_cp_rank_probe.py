"""Bounds for the CP rank of the reduced IEEE-9 coefficient tensor.

For 49 reduced equations in 49 states, the conceptual tensor has shape
``(49, 2, ..., 2)``. Its nonzero coefficients come from the fast S/Phi
extractor; the exponentially large tensor is never allocated. Numerically
measured matrix ranks of several unfoldings bound its CP rank from below.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from collections import Counter

from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_gfl_ode_reduced import build_problem


def unfolding_rank(phi: sparse.csr_matrix, monomials: list[tuple[int, ...]],
                   left_modes: set[int]) -> tuple[tuple[int, int], int]:
    """Put the equation mode and ``left_modes`` on one side of a matrix."""
    row_indices: dict[tuple[int, tuple[int, ...]], int] = {}
    column_indices: dict[tuple[int, ...], int] = {}
    rows, columns, values = [], [], []
    for equation in range(phi.shape[0]):
        for pointer in range(phi.indptr[equation], phi.indptr[equation + 1]):
            monomial = monomials[int(phi.indices[pointer])]
            left = tuple(index for index in monomial if index in left_modes)
            right = tuple(index for index in monomial if index not in left_modes)
            rows.append(row_indices.setdefault((equation, left), len(row_indices)))
            columns.append(column_indices.setdefault(right, len(column_indices)))
            values.append(float(phi.data[pointer]))
    unfolding = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(row_indices), len(column_indices)),
    )
    return unfolding.shape, int(np.linalg.matrix_rank(unfolding.toarray()))


def main() -> None:
    reduced = build_problem().build_reduced_model_problem()
    state_count = len(reduced.get_state_vars())
    S, Phi = reduced.build_multilinear_matrices_fast()
    monomials = [
        tuple(int(index) for index in S.indices[S.indptr[column]:S.indptr[column + 1]]
              if index < state_count)
        for column in range(S.shape[1])
    ]
    print(f"coefficient tensor: ({Phi.shape[0]}, 2^{state_count}), "
          f"nonzero monomials={len(monomials)}, coefficient entries={Phi.nnz}")
    degrees = Counter(map(len, monomials))
    entries_by_degree = Counter()
    for column, monomial in enumerate(monomials):
        entries_by_degree[len(monomial)] += Phi.getcol(column).nnz
    print(f"monomial supports by degree={dict(sorted(degrees.items()))}")
    print(f"coefficient entries by degree={dict(sorted(entries_by_degree.items()))}")
    partitions = {
        "source and first converter": set(range(23)),
        "alternating states": set(range(0, state_count, 2)),
        "first 12 states": set(range(12)),
    }
    for label, left_modes in partitions.items():
        shape, rank = unfolding_rank(Phi, monomials, left_modes)
        print(f"{label}: unfolding={shape}, CP rank >= {rank}")
    print(f"monomial CP construction: CP rank <= {len(monomials)}")

    # A basic exact CP compression would merge two monomial terms that differ
    # in one state mode when their 49-dimensional output factors are collinear.
    # Their binary mode factors then combine into one affine [a, b] factor.
    # This is only a first-order merge test, not a general CP-rank algorithm.
    dense_outputs = Phi.toarray()
    column_by_monomial = {frozenset(m): j for j, m in enumerate(monomials)}
    candidate_pairs = mergeable_pairs = 0
    for j, monomial in enumerate(monomials):
        output = dense_outputs[:, j]
        output_norm = np.linalg.norm(output)
        for mode in monomial:
            other = column_by_monomial.get(frozenset(monomial) - {mode})
            if other is None:
                continue
            other_output = dense_outputs[:, other]
            other_norm = np.linalg.norm(other_output)
            candidate_pairs += 1
            cosine_error = np.linalg.norm(
                output / output_norm
                - np.sign(output @ other_output) * other_output / other_norm
            )
            mergeable_pairs += cosine_error < 1e-12
    print(f"one-mode CP merge candidates={candidate_pairs}, "
          f"collinear output pairs={mergeable_pairs}")


if __name__ == "__main__":
    main()
