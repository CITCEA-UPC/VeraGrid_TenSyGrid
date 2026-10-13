"""Exact grouped tensor decomposition of the degree-five IEEE-9 equations.

Only the highest-degree terms of the two GFL converters are studied here.
Each such term chooses one state from five disjoint physical groups: network
source/current, converter line current, and three two-state angle pairs.
The output equations form a sixth mode. This gives a small, genuine
multilinear coefficient tensor for each converter, suitable for TT-SVD.

The grouped modes are *not* the 49 binary monomial modes of the full
Lichtenberg CP tensor. Consequently these TT ranks are not CP ranks of that
full tensor. The decomposition is an exact factored evaluator for this block.
"""

from __future__ import annotations

import numpy as np

from TensyGridEngine.emt.emt_explicit_converter_demo.ieee9_gfl_ode_reduced import build_problem


NETWORK = (0, 1, 2, 20, 21, 22, 37, 38, 39, *range(40, 49))
CONVERTERS = (
    ((6, 7, 8, 20, 21, 22), ((6, 7, 8), (14, 15), (16, 17), (18, 19))),
    ((23, 24, 25, 37, 38, 39), ((23, 24, 25), (31, 32), (33, 34), (35, 36))),
)


def degree_five_block(phi, monomials, rows, groups):
    """Gather coefficients into (equation, group 1, ..., group 5)."""
    all_groups = (NETWORK, *groups)
    block = np.zeros((len(rows), *(len(group) for group in all_groups)))
    row_positions = {row: position for position, row in enumerate(rows)}
    used = set()
    for row in rows:
        for pointer in range(phi.indptr[row], phi.indptr[row + 1]):
            column = int(phi.indices[pointer])
            monomial = monomials[column]
            if len(monomial) != 5:
                continue
            coordinates = []
            for group in all_groups:
                matching = [position for position, state in enumerate(group)
                            if state in monomial]
                if len(matching) != 1:
                    raise ValueError(f"Degree-five monomial {monomial} breaks grouping")
                coordinates.append(matching[0])
            block[(row_positions[row], *coordinates)] = phi.data[pointer]
            used.add(column)
    return block, len(used)


def tt_svd(tensor: np.ndarray, relative_tolerance: float = 1e-12):
    """Compute a real TT by SVD, discarding only numerical zero singular values."""
    cores = []
    unfolding = tensor.copy()
    left_rank = 1
    ranks = []
    for mode_size in tensor.shape[:-1]:
        matrix = unfolding.reshape(left_rank * mode_size, -1)
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        rank = int(np.count_nonzero(singular > relative_tolerance * singular[0]))
        cores.append(left[:, :rank].reshape(left_rank, mode_size, rank))
        unfolding = singular[:rank, None] * right[:rank]
        left_rank = rank
        ranks.append(rank)
    cores.append(unfolding.reshape(left_rank, tensor.shape[-1], 1))
    return cores, ranks


def evaluate_tt(cores, grouped_values):
    """Contract the five state groups; leave the equation mode as output."""
    output = cores[0][0]
    for core, values in zip(cores[1:], grouped_values):
        output = output @ np.einsum("aib,i->ab", core, values)
    return output.ravel()


def main() -> None:
    reduced = build_problem().build_reduced_model_problem()
    S, phi = reduced.build_multilinear_matrices_fast()
    state_count = len(reduced.get_state_vars())
    monomials = [
        tuple(int(index) for index in S.indices[S.indptr[column]:S.indptr[column + 1]]
              if index < state_count)
        for column in range(S.shape[1])
    ]
    rng = np.random.default_rng(20261009)
    for converter, (rows, groups) in enumerate(CONVERTERS, start=1):
        block, support_count = degree_five_block(phi, monomials, rows, groups)
        cores, ranks = tt_svd(block)
        reconstructed = cores[0][0]
        for core in cores[1:]:
            reconstructed = np.tensordot(reconstructed, core, axes=(-1, 0))
        reconstructed = reconstructed[..., 0]
        coefficient_error = np.linalg.norm(reconstructed - block) / np.linalg.norm(block)

        evaluation_errors = []
        for _ in range(20):
            states = rng.standard_normal(state_count) * 0.1
            values = [states[list(group)] for group in (NETWORK, *groups)]
            direct = np.einsum("eabcdf,a,b,c,d,f->e", block, *values)
            factored = evaluate_tt(cores, values)
            evaluation_errors.append(
                np.linalg.norm(factored - direct) / max(np.linalg.norm(direct), 1e-14)
            )
        stored_values = sum(core.size for core in cores)
        if coefficient_error > 1e-11 or max(evaluation_errors) > 1e-10:
            raise RuntimeError(f"Converter {converter} TT reconstruction failed")
        print(f"converter {converter}: block shape={block.shape}, "
              f"degree-five supports={support_count}, coefficient entries={np.count_nonzero(block)}")
        print(f"  TT ranks={ranks}, stored floats={stored_values}, "
              f"dense floats={block.size}")
        print(f"  coefficient error={coefficient_error:.3e}, "
              f"max evaluation error={max(evaluation_errors):.3e}")


if __name__ == "__main__":
    main()
