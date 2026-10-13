"""Accuracy and speed of approximate grouped CP for complete converters.

Fits all polynomial degrees of each 17-output converter at once. Coefficient
error includes absent monomials, so a fit cannot hide spurious terms. Reports
RHS errors near the initialized IEEE-9 state and for small random states,
then compares repeated CP and sparse S/Phi evaluations. An optional short
Radau trajectory comparison uses the same integrator for exact and approximate
right-hand sides; it is not the production EMT solver.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from numba import njit

from TensyGridEngine.emt.emt_explicit_converter_demo.ieee9_gfl_ode_reduced import build_problem
from TensyGridEngine.emt.emt_explicit_converter_demo.tensor_decompositions.ieee9_gfl_ode_full_grouped_tensor import (
    CONVERTERS, NETWORK, full_converter_tensor, grouped_values,
)


def evaluate_cp(weights, factors, values):
    """Evaluate one complete converter's grouped CP expression."""
    products = weights.copy()
    for factor, value in zip(factors[1:], values):
        products *= value @ factor
    return factors[0] @ products


def timed_evaluation(fn, states):
    """Best of three serial passes, in microseconds per state vector."""
    for x in states[:10]:
        fn(x)
    timings = []
    for _ in range(3):
        start = perf_counter()
        for x in states:
            fn(x)
        timings.append((perf_counter() - start) * 1e6 / len(states))
    return min(timings)


@njit(cache=True)
def compiled_cp_eval(x, weights, output_factors, packed_factors, packed_indices,
                     family_lengths):
    """Evaluate grouped CP with scalar loops and no small NumPy dispatches."""
    result = np.zeros(output_factors.shape[0])
    for term in range(weights.size):
        product = weights[term]
        for family in range(6):
            linear_form = packed_factors[family, 0, term]
            for j in range(family_lengths[family]):
                linear_form += packed_factors[family, j + 1, term] * x[packed_indices[family, j]]
            product *= linear_form
        for row in range(output_factors.shape[0]):
            result[row] += output_factors[row, term] * product
    return result


@njit(cache=True)
def compiled_sparse_eval(x, csc_indptr, csc_indices, csc_data, padded, lengths,
                         output_count):
    """Evaluate the same sparse monomials using compiled scalar loops."""
    result = np.zeros(output_count)
    for column in range(len(lengths)):
        product = 1.0
        for j in range(lengths[column]):
            product *= x[padded[column, j]]
        for pointer in range(csc_indptr[column], csc_indptr[column + 1]):
            result[csc_indices[pointer]] += csc_data[pointer] * product
    return result


def main(rank: int = 80, iterations: int = 250, samples: int = 400,
         compiled: bool = False, trajectory: bool = False) -> None:
    reduced = build_problem().build_reduced_model_problem()
    S, phi = reduced.build_multilinear_matrices_fast()
    monomials = [
        tuple(int(i) for i in S.indices[S.indptr[j]:S.indptr[j + 1]])
        for j in range(S.shape[1])
    ]
    initial = reduced.get_x0()
    rng = np.random.default_rng(20261009)
    small_states = rng.standard_normal((samples, 49)) * 0.1
    near_initial = initial + rng.standard_normal((samples, 49)) * (0.01 * np.maximum(1.0, np.abs(initial)))
    print(f"initial state range=[{initial.min():.3e}, {initial.max():.3e}]", flush=True)
    compiled_models = []

    for converter, (rows, line, voltage, current, output, controller) in enumerate(CONVERTERS, 1):
        families = (NETWORK, line, voltage, current, output, controller)
        tensor = full_converter_tensor(phi, monomials, rows, families)
        start = perf_counter()
        cp = parafac(tensor, rank=rank, n_iter_max=iterations, tol=1e-12,
                     init="random", random_state=1, normalize_factors=True)
        fit_seconds = perf_counter() - start
        coefficient_error = np.linalg.norm(tl.cp_to_tensor(cp) - tensor) / np.linalg.norm(tensor)
        weights, factors = cp

        sparse_rows = phi[list(rows)].tocsr()
        used = np.unique(sparse_rows.indices)
        sparse_rows = sparse_rows[:, used].tocsr()
        padded = np.full((len(used), 5), 49, dtype=np.intp)
        for position, column in enumerate(used):
            monomial = monomials[int(column)]
            padded[position, :len(monomial)] = monomial

        def sparse_eval(x):
            augmented = np.empty(50)
            augmented[:49] = x
            augmented[49] = 1.0
            return sparse_rows @ np.prod(augmented[padded], axis=1)

        def cp_eval(x):
            return evaluate_cp(weights, factors, grouped_values(x, families))

        def errors(states):
            relative = []
            absolute = []
            for x in states:
                reference = sparse_eval(x)
                difference = np.linalg.norm(cp_eval(x) - reference)
                absolute.append(difference)
                relative.append(difference / max(np.linalg.norm(reference), 1e-14))
            return float(np.median(relative)), float(np.max(relative)), float(np.max(absolute))

        initial_error = errors(initial[None, :])
        small_error = errors(small_states)
        nearby_error = errors(near_initial)
        sparse_time = timed_evaluation(sparse_eval, near_initial)
        cp_time = timed_evaluation(cp_eval, near_initial)
        stored = weights.size + sum(factor.size for factor in factors)
        print(f"converter {converter}: rank={rank}, coefficient error={coefficient_error:.3e}, "
              f"stored floats={stored}, fit seconds={fit_seconds:.1f}", flush=True)
        print(f"  initial RHS relative/absolute error={initial_error[1]:.3e}/{initial_error[2]:.3e}",
              flush=True)
        print(f"  small-state RHS relative median/max={small_error[0]:.3e}/{small_error[1]:.3e}",
              flush=True)
        print(f"  near-initial RHS relative median/max={nearby_error[0]:.3e}/{nearby_error[1]:.3e}",
              flush=True)
        print(f"  sparse={sparse_time:.2f} us/call, CP={cp_time:.2f} us/call, "
              f"CP/sparse={cp_time/sparse_time:.2f}x", flush=True)

        if compiled or trajectory:
            packed_factors = np.zeros((6, 22, rank))
            packed_indices = np.zeros((6, 21), dtype=np.int64)
            family_lengths = np.array([len(family) for family in families], dtype=np.int64)
            for family_number, (family, factor) in enumerate(zip(families, factors[1:])):
                packed_factors[family_number, :factor.shape[0], :] = factor
                packed_indices[family_number, :len(family)] = family
            csc = sparse_rows.tocsc()
            lengths = np.array([len(monomials[int(column)]) for column in used], dtype=np.int64)
            def compiled_cp(x):
                return compiled_cp_eval(x, weights, factors[0], packed_factors,
                                        packed_indices, family_lengths)
            def compiled_sparse(x):
                return compiled_sparse_eval(x, csc.indptr, csc.indices, csc.data,
                                            padded, lengths, len(rows))
            compiled_cp(initial)
            compiled_sparse(initial)
            compiled_error = max(
                np.linalg.norm(compiled_cp(x) - cp_eval(x))
                / max(np.linalg.norm(cp_eval(x)), 1e-14)
                for x in near_initial[:20]
            )
            if compiled_error > 1e-10:
                raise RuntimeError(f"Compiled CP differs from NumPy CP: {compiled_error}")
            sparse_jit_time = timed_evaluation(compiled_sparse, near_initial)
            cp_jit_time = timed_evaluation(compiled_cp, near_initial)
            print(f"  compiled sparse={sparse_jit_time:.2f} us/call, "
                  f"compiled CP={cp_jit_time:.2f} us/call, "
                  f"CP/sparse={cp_jit_time/sparse_jit_time:.2f}x", flush=True)
            compiled_models.append((weights, factors[0], packed_factors,
                                    packed_indices, family_lengths))

    if trajectory:
        from scipy.integrate import solve_ivp

        full_csc = phi.tocsc()
        full_padded = np.full((len(monomials), 5), 49, dtype=np.intp)
        full_lengths = np.array([len(monomial) for monomial in monomials], dtype=np.int64)
        for column, monomial in enumerate(monomials):
            full_padded[column, :len(monomial)] = monomial

        other_rows = np.array([*range(6), *range(40, 49)])
        other_sparse = phi[other_rows].tocsr()
        other_used = np.unique(other_sparse.indices)
        other_csc = other_sparse[:, other_used].tocsc()
        other_padded = np.full((len(other_used), 5), 49, dtype=np.intp)
        other_lengths = np.array([len(monomials[int(column)]) for column in other_used],
                                 dtype=np.int64)
        for position, column in enumerate(other_used):
            monomial = monomials[int(column)]
            other_padded[position, :len(monomial)] = monomial

        def exact_rhs(_time, x):
            return compiled_sparse_eval(x, full_csc.indptr, full_csc.indices,
                                        full_csc.data, full_padded, full_lengths, 49)

        def approximate_rhs(time, x):
            result = np.empty(49)
            other_values = compiled_sparse_eval(x, other_csc.indptr, other_csc.indices,
                                                other_csc.data, other_padded,
                                                other_lengths, len(other_rows))
            result[other_rows] = other_values
            for rows, model in zip(CONVERTERS, compiled_models):
                result[list(rows[0])] = compiled_cp_eval(x, *model)
            return result

        sample_times = np.linspace(0.0, 1.0e-4, 6)
        exact_rhs(0.0, initial)
        approximate_rhs(0.0, initial)
        exact_times = []
        approximate_times = []
        for _ in range(3):
            start = perf_counter()
            exact_solution = solve_ivp(exact_rhs, (sample_times[0], sample_times[-1]),
                                       initial, method="Radau", t_eval=sample_times,
                                       rtol=1e-9, atol=1e-10)
            exact_times.append(perf_counter() - start)
            start = perf_counter()
            approximate_solution = solve_ivp(approximate_rhs,
                                             (sample_times[0], sample_times[-1]),
                                             initial, method="Radau", t_eval=sample_times,
                                             rtol=1e-9, atol=1e-10)
            approximate_times.append(perf_counter() - start)
        if not exact_solution.success or not approximate_solution.success:
            raise RuntimeError("Exact or approximate short trajectory failed")
        differences = approximate_solution.y - exact_solution.y
        print(f"short trajectory: 0..{sample_times[-1]:.1e} s, "
              f"max state absolute error={np.max(np.abs(differences)):.3e}, "
              f"final relative state error={np.linalg.norm(differences[:, -1])/np.linalg.norm(exact_solution.y[:, -1]):.3e}",
              flush=True)
        print(f"  Radau RHS calls: exact={exact_solution.nfev}, "
              f"approximate={approximate_solution.nfev}", flush=True)
        print(f"  Radau wall seconds (best of 3): exact={min(exact_times):.4f}, "
              f"approximate={min(approximate_times):.4f}, "
              f"approx/exact={min(approximate_times)/min(exact_times):.2f}x",
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--trajectory", action="store_true",
                        help="Compare short Radau trajectories after fitting both converters")
    arguments = parser.parse_args()
    main(rank=arguments.rank, iterations=arguments.iterations,
         samples=arguments.samples, compiled=arguments.compiled,
         trajectory=arguments.trajectory)
