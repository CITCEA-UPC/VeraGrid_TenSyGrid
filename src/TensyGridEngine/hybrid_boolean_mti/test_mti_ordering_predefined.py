from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from VeraGridEngine.Simulations.Rms.problems.mti_hybrid_structure import (
    _order_incidence_matrix_dmperm_like,
    build_connected_subproblem_order,
)


def summarize(
    name: str,
    incidence: np.ndarray,
    expected_components: int,
    expected_isorted: np.ndarray | None = None,
) -> None:
    print(f"\n{name}")
    print(incidence.astype(int))
    isorted, row_sort, col_sort = _order_incidence_matrix_dmperm_like(incidence)
    print("Isorted")
    print(isorted.astype(int))
    permutation_ok = _is_permutation(row_sort, incidence.shape[0]) and _is_permutation(col_sort, incidence.shape[1])
    reconstruction_ok = np.array_equal(isorted, incidence[np.ix_(row_sort, col_sort)])
    expected_isorted_ok = expected_isorted is None or np.array_equal(isorted, expected_isorted)
    order = build_connected_subproblem_order(incidence)
    subsets = sorted({int(r.subset) for r in order})
    subproblems = sorted({int(r.subproblem) for r in order})
    explicit = int(sum(int(r.explicit) for r in order))
    components_ok = len(subproblems) == expected_components
    transform_ok = _has_no_cross_block_coupling(isorted, order)
    print(
        f"summary rows={len(order)} subsets={len(subsets)} "
        f"subproblems={len(subproblems)} explicit={explicit}"
    )
    print(
        f"checks components_ok={components_ok} transform_ok={transform_ok} "
        f"permutation_ok={permutation_ok} reconstruction_ok={reconstruction_ok} "
        f"expected_isorted_ok={expected_isorted_ok}"
    )
    print("order", [(r.eq_idx, r.var_idx, r.subset, r.subproblem, r.explicit) for r in order])
    if not (components_ok and transform_ok and permutation_ok and reconstruction_ok and expected_isorted_ok):
        raise AssertionError(
            f"{name} failed: components_ok={components_ok}, transform_ok={transform_ok}, "
            f"permutation_ok={permutation_ok}, reconstruction_ok={reconstruction_ok}, "
            f"expected_isorted_ok={expected_isorted_ok}"
        )


def _is_permutation(values: np.ndarray, n: int) -> bool:
    return values.size == n and np.array_equal(np.sort(values), np.arange(n, dtype=int))


def _has_no_cross_block_coupling(isorted: np.ndarray, order: list) -> bool:
    """Check that MATLAB-style subproblem cuts have no upper-right coupling."""
    n = isorted.shape[0]
    subproblem = np.asarray([int(r.subproblem) for r in order], dtype=int)
    for l in range(1, n):
        if subproblem[l] == subproblem[l - 1]:
            continue
        if np.any(isorted[:l, l:]):
            return False
    return True


def main() -> None:
    # Fully separable: each equation depends on exactly one variable.
    summarize(
        "diagonal_4",
        np.eye(4, dtype=int),
        expected_components=4,
        expected_isorted=np.eye(4, dtype=int),
    )

    # Two independent 2x2 irreducible blocks.
    summarize(
        "block_diag_two_2x2",
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=int,
        ),
        expected_components=2,
        expected_isorted=np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=int,
        ),
    )

    # Lower block triangular with two 2x2 diagonal blocks and downstream coupling.
    summarize(
        "lower_block_triangular",
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
            ],
            dtype=int,
        ),
        expected_components=2,
        expected_isorted=np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
            ],
            dtype=int,
        ),
    )

    # Upper block triangular counterpart.
    summarize(
        "upper_block_triangular",
        np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=int,
        ),
        expected_components=2,
        expected_isorted=np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
            ],
            dtype=int,
        ),
    )

    # One irreducible cycle: should remain one block.
    summarize(
        "irreducible_cycle_4",
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 0, 1],
            ],
            dtype=int,
        ),
        expected_components=1,
        expected_isorted=np.array(
            [
                [1, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=int,
        ),
    )


if __name__ == "__main__":
    main()
