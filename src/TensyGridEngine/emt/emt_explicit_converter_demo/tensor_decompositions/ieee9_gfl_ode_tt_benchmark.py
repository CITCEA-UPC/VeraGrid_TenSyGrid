"""Compare full-converter RHS evaluation using sparse S/Phi and exact TT.

The build/factorization time is reported separately. Both evaluators consume
the same 49-state vectors and return the same 17 derivative values. This is a
microbenchmark of coefficient evaluators, not a full EMT solver benchmark.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_gfl_ode_reduced import build_problem
from TensyGridEngine.emt.emt_explicit_converter_demo.tensor_decompositions.ieee9_gfl_ode_degree5_tt import (
    evaluate_tt, tt_svd,
)
from TensyGridEngine.emt.emt_explicit_converter_demo.tensor_decompositions.ieee9_gfl_ode_full_grouped_tensor import (
    CONVERTERS, NETWORK, full_converter_tensor, grouped_values,
)


def benchmark(samples: int = 400) -> None:
    start = perf_counter()
    reduced = build_problem().build_reduced_model_problem()
    S, phi = reduced.build_multilinear_matrices_fast()
    extraction_seconds = perf_counter() - start
    monomials = [
        tuple(int(i) for i in S.indices[S.indptr[j]:S.indptr[j + 1]])
        for j in range(S.shape[1])
    ]
    states = np.random.default_rng(42).normal(size=(samples, 49)) * 0.1

    print(f"model and S/Phi extraction seconds={extraction_seconds:.3f}")
    for converter, (rows, line, voltage, current, output, controller) in enumerate(CONVERTERS, 1):
        families = (NETWORK, line, voltage, current, output, controller)
        tensor = full_converter_tensor(phi, monomials, rows, families)
        start = perf_counter()
        cores, _ = tt_svd(tensor)
        factor_seconds = perf_counter() - start

        sparse_rows = phi[list(rows)].tocsr()
        used_columns = np.unique(sparse_rows.indices)
        sparse_rows = sparse_rows[:, used_columns].tocsr()
        padded = np.full((len(used_columns), 5), 49, dtype=np.intp)
        for position, column in enumerate(used_columns):
            monomial = monomials[int(column)]
            padded[position, :len(monomial)] = monomial

        def sparse_eval(x):
            augmented = np.empty(50)
            augmented[:49] = x
            augmented[49] = 1.0
            return sparse_rows @ np.prod(augmented[padded], axis=1)

        def tt_eval(x):
            return evaluate_tt(cores, grouped_values(x, families))

        relative_errors = []
        for x in states[:20]:
            reference = sparse_eval(x)
            relative_errors.append(np.linalg.norm(tt_eval(x) - reference)
                                   / max(np.linalg.norm(reference), 1e-14))
        if max(relative_errors) > 1e-10:
            raise RuntimeError(f"Converter {converter} TT evaluation differs from sparse S/Phi")

        def microseconds_per_call(fn):
            for x in states[:10]:
                fn(x)
            repeats = []
            for _ in range(3):
                start = perf_counter()
                for x in states:
                    fn(x)
                repeats.append((perf_counter() - start) * 1e6 / len(states))
            return min(repeats)

        sparse_time = microseconds_per_call(sparse_eval)
        tt_time = microseconds_per_call(tt_eval)
        print(f"converter {converter}: sparse={sparse_time:.2f} us/call, "
              f"TT={tt_time:.2f} us/call, TT/sparse={tt_time/sparse_time:.2f}x")
        print(f"  TT factorization seconds={factor_seconds:.3f}, "
              f"max relative evaluation error={max(relative_errors):.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=400)
    benchmark(samples=parser.parse_args().samples)
