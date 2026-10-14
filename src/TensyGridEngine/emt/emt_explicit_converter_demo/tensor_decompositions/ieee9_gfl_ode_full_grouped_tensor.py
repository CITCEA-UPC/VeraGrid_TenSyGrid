"""Exact full-converter tensor grouped by disjoint physical state families.

Each of the 17 converter equations includes all polynomial degrees. In this
example every monomial selects at most one state from each of six disjoint
families. Adding a constant coordinate to every family therefore encodes the
entire polynomial, including absent factors, in one modest grouped tensor.

This is a useful intermediate representation, but its modes are *not* the
49 binary modes of Lichtenberg's original coefficient tensor. A grouped
TT decomposition does not determine that tensor's CP rank.
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from TensyGridEngine.emt.emt_explicit_converter_demo.tensor_decompositions.ieee9_gfl_ode_degree5_tt import (
    tt_svd, evaluate_tt,
)
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_gfl_ode_reduced import build_problem


NETWORK = (0, 1, 2, 3, 4, 5, *range(20, 23), *range(37, 49))
CONVERTERS = (
    (tuple(range(6, 23)), (6, 7, 8), (14, 15), (16, 17), (18, 19), tuple(range(9, 14))),
    (tuple(range(23, 40)), (23, 24, 25), (31, 32), (33, 34), (35, 36), tuple(range(26, 31))),
)


def full_converter_tensor(phi, monomials, rows, families):
    """Gather all coefficients; coordinate zero means no state from a family."""
    shape = (len(rows), *(1 + len(family) for family in families))
    tensor = np.zeros(shape)
    for output, row in enumerate(rows):
        for pointer in range(phi.indptr[row], phi.indptr[row + 1]):
            monomial = set(monomials[int(phi.indices[pointer])])
            coordinates = []
            for family in families:
                matches = [j + 1 for j, state in enumerate(family) if state in monomial]
                if len(matches) > 1:
                    raise ValueError(f"Monomial {monomial} uses multiple states in one family")
                coordinates.append(matches[0] if matches else 0)
            if sum(coordinate != 0 for coordinate in coordinates) != len(monomial):
                raise ValueError(f"Monomial {monomial} contains an ungrouped state")
            tensor[(output, *coordinates)] = phi.data[pointer]
    return tensor


def grouped_values(states, families):
    """Prepend the constant coordinate to each group of state values."""
    return [np.r_[1.0, states[list(family)]] for family in families]


def main(*, cp_ranks: tuple[int, ...] = (), cp_iterations: int = 100,
         converter_filter: int | None = None) -> None:
    reduced = build_problem().build_reduced_model_problem()
    S, phi = reduced.build_multilinear_matrices_fast()
    n_state = len(reduced.get_state_vars())
    monomials = [
        tuple(int(index) for index in S.indices[S.indptr[column]:S.indptr[column + 1]]
              if index < n_state)
        for column in range(S.shape[1])
    ]
    rng = np.random.default_rng(20261009)
    for converter, (rows, line, voltage, current, output, controller) in enumerate(CONVERTERS, 1):
        if converter_filter is not None and converter != converter_filter:
            continue
        families = (NETWORK, line, voltage, current, output, controller)
        tensor = full_converter_tensor(phi, monomials, rows, families)
        cores, ranks = tt_svd(tensor)
        reconstructed = cores[0][0]
        for core in cores[1:]:
            reconstructed = np.tensordot(reconstructed, core, axes=(-1, 0))
        coefficient_error = np.linalg.norm(reconstructed[..., 0] - tensor) / np.linalg.norm(tensor)
        evaluation_errors = []
        for _ in range(20):
            states = rng.standard_normal(n_state) * 0.1
            values = grouped_values(states, families)
            direct = np.einsum("zabcdef,a,b,c,d,e,f->z", tensor, *values)
            factored = evaluate_tt(cores, values)
            evaluation_errors.append(np.linalg.norm(factored - direct)
                                     / max(np.linalg.norm(direct), 1e-14))
        if coefficient_error > 1e-11 or max(evaluation_errors) > 1e-10:
            raise RuntimeError(f"Converter {converter} full grouped TT failed")
        print(f"converter {converter}: full grouped shape={tensor.shape}, "
              f"nonzero coefficients={np.count_nonzero(tensor)}")
        print(f"  TT ranks={ranks}, stored floats={sum(core.size for core in cores)}, "
              f"dense floats={tensor.size}")
        print(f"  coefficient error={coefficient_error:.3e}, "
              f"max evaluation error={max(evaluation_errors):.3e}")

        if cp_ranks:
            import tensorly as tl
            from tensorly.decomposition import parafac

            for rank in cp_ranks:
                start = time.perf_counter()
                cp = parafac(
                    tensor, rank=rank, n_iter_max=cp_iterations, tol=1e-12,
                    init="random", random_state=1, normalize_factors=True,
                )
                cp_error = np.linalg.norm(tl.cp_to_tensor(cp) - tensor) / np.linalg.norm(tensor)
                weights, factors = cp
                stored = weights.size + sum(factor.size for factor in factors)
                print(f"  grouped CP rank={rank}: coefficient error={cp_error:.3e}, "
                      f"stored floats={stored}, fit seconds={time.perf_counter()-start:.1f}",
                      flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-ranks", nargs="*", type=int, default=(),
                        help="Optional grouped CP-ALS ranks to try")
    parser.add_argument("--cp-iterations", type=int, default=100)
    parser.add_argument("--converter", type=int, choices=(1, 2))
    arguments = parser.parse_args()
    main(cp_ranks=tuple(arguments.cp_ranks), cp_iterations=arguments.cp_iterations,
         converter_filter=arguments.converter)
