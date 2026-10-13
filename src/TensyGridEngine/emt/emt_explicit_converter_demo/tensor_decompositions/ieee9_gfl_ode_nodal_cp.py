"""CP representation of the 27-input IEEE-9 nodal inverse.

For v = G^{-1} i, the multilinear coefficient tensor has one output mode
of size 27 and 27 binary modes, one for each component of i. A binary mode
selects the constant or first-degree coefficient of that input. The huge
27-by-2**27 tensor is never allocated: its only nonzero columns are the 27
degree-one monomials, and those columns are exactly G^{-1}.

This is the CP tensor used for multilinear polynomial equations. It differs
from the bus-pair/phase-pair unfolding in ``ieee9_gfl_ode_nodal_tensor``.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from time import perf_counter

def evaluate_monomial_cp(inverse: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Evaluate the exact 27-term CP sum without forming its 27 binary modes.

    Term j has output factor ``inverse[:, j]``. Its input factor is [0, 1]
    in mode j and [1, 0] in every other mode. Contracting the modes with
    [1, current[k]] therefore gives exactly ``inverse[:, j] * current[j]``.
    """
    if inverse.shape != (27, 27) or current.shape != (27,):
        raise ValueError("Expected a 27-by-27 inverse and 27 input currents")
    result = np.zeros(27)
    for j in range(27):
        result += inverse[:, j] * current[j]
    return result


def phase_factors(inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract the two 9-by-9 bus factors from the balanced nodal inverse."""
    tensor = inverse.reshape(9, 3, 9, 3).transpose(0, 2, 1, 3)
    common = np.ones((3, 3)) / 3.0
    differential = np.eye(3) - common
    bus_common = np.einsum("abpq,pq->ab", tensor, common)
    bus_differential = np.einsum("abpq,pq->ab", tensor, differential) / 2.0
    return bus_common, bus_differential


def structured_coefficients(bus_common: np.ndarray,
                            bus_differential: np.ndarray) -> np.ndarray:
    """Generate the 27 CP output factors from two bus factors on demand."""
    common = np.ones((3, 3)) / 3.0
    differential = np.eye(3) - common
    tensor = (np.einsum("ab,pq->apbq", bus_common, common)
              + np.einsum("ab,pq->apbq", bus_differential, differential))
    return tensor.reshape(27, 27)


def evaluate_structured(bus_common: np.ndarray, bus_differential: np.ndarray,
                        current: np.ndarray) -> np.ndarray:
    """Contract all 27 CP terms through the two phase projectors."""
    phases = current.reshape(9, 3)
    mean = phases.mean(axis=1, keepdims=True)
    return (bus_common @ mean + bus_differential @ (phases - mean)).ravel()


def timed_call(fn, current: np.ndarray, repetitions: int = 5000) -> float:
    """Best of three serial timings, in microseconds per evaluation."""
    fn(current)
    elapsed = []
    for _ in range(3):
        start = perf_counter()
        for _ in range(repetitions):
            fn(current)
        elapsed.append(perf_counter() - start)
    return min(elapsed) * 1e6 / repetitions


def main() -> None:
    from TensyGridEngine.emt.emt_explicit_converter_demo.tensor_decompositions.ieee9_gfl_ode_nodal_tensor import nodal_matrix
    from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_gfl_ode_reduced import build_problem

    conductance = nodal_matrix(build_problem())
    inverse = linalg.inv(conductance)
    bus_common, bus_differential = phase_factors(inverse)
    generated = structured_coefficients(bus_common, bus_differential)
    coefficient_error = np.linalg.norm(generated - inverse) / np.linalg.norm(inverse)
    singular_values = linalg.svdvals(inverse)

    # The output-mode unfolding is a 27-by-2**27 matrix with just 27
    # nonzero columns. Removing its zero columns leaves G^{-1}. Any CP
    # decomposition has at least its matrix rank, hence at least 27 terms.
    # The monomial CP construction above attains that bound exactly.
    unfolding_rank = np.linalg.matrix_rank(inverse)
    if unfolding_rank != 27:
        raise RuntimeError("The nodal inverse is unexpectedly singular")

    rng = np.random.default_rng(20261008)
    errors = []
    structured_errors = []
    for _ in range(20):
        current = rng.standard_normal(27)
        direct = linalg.solve(conductance, current)
        cp_result = evaluate_monomial_cp(inverse, current)
        errors.append(np.linalg.norm(direct - cp_result) / np.linalg.norm(direct))
        structured_result = evaluate_structured(bus_common, bus_differential, current)
        structured_errors.append(np.linalg.norm(direct - structured_result) / np.linalg.norm(direct))
    max_error = max(errors)
    max_structured_error = max(structured_errors)
    if max(max_error, max_structured_error, coefficient_error) > 1e-12:
        raise RuntimeError("CP or structured reconstruction differs from the nodal solve")

    current = rng.standard_normal(27)
    dense_time = timed_call(lambda x: inverse @ x, current)
    cp_time = timed_call(lambda x: evaluate_monomial_cp(inverse, x), current)
    structured_time = timed_call(
        lambda x: evaluate_structured(bus_common, bus_differential, x), current
    )

    print("Coefficient tensor shape=(27,) + (2,)*27 (kept implicit)")
    print(f"nonzero monomial columns=27, output unfolding rank={unfolding_rank}")
    print(f"exact CP rank={unfolding_rank} (matching lower and upper bounds)")
    print(f"inverse singular-value range={singular_values[-1]:.3e}..{singular_values[0]:.3e}")
    print(f"max relative solve error over 20 currents={max_error:.3e}")
    print(f"structured coefficient reconstruction error={coefficient_error:.3e}")
    print(f"structured max relative solve error={max_structured_error:.3e}")
    print("stored floats: dense inverse=729, explicit CP output factors=729, "
          "structured bus factors=162")
    print(f"evaluation microseconds: dense={dense_time:.2f}, "
          f"explicit CP={cp_time:.2f}, structured={structured_time:.2f}")
    print("bus/phase Kronecker separation rank=2 (different tensorization)")


if __name__ == "__main__":
    main()
