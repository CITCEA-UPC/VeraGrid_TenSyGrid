"""Extract the full polynomial of the charged IEEE-9 explicit vector field.

The 103 reduced coordinates are the 76 non-charge states followed by 27 bus
voltages. Sparse monomial maps keep repeated coordinates, so a squared state
cannot silently pass a multi-affinity check.
"""

from collections import Counter

import numpy as np
from scipy import linalg

from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_gfl_ode_reduced import build_problem
from TensyGridEngine.emt.emt_explicit_converter_demo.experiments.ieee9_gfl_ode_pi_charge_ode import charge_reduction_data


def add(poly, term, value):
    poly[term] = poly.get(term, 0.0) + value


def multiply(left, right):
    product = {}
    for a, av in left.items():
        for b, bv in right.items():
            add(product, tuple(sorted(a + b)), av * bv)
    return product


def extract(*, replacement_B=0.1, first_gfl_power_scale=1.0,
            symbolic_first_gfl_reference=False, symbolic_references=()):
    """Assemble the grid DAE, then eliminate it into polynomial ODE rows.

    This is the indirect ``build_problem`` call used by ieee9_full_explicit.py.
    ``output[i]`` maps a tuple of reduced-state indices to its coefficient in
    dz_i/dt. For example, ``(2, 7): c`` means ``c*z_2*z_7``. When the first
    GFL reference stays symbolic, mode 103 is its P_ref input.
    """
    # The original device equations live in the EMT templates. build_problem
    # attaches them to the MultiCircuit and assembles the full grid DAE.
    problem = build_problem(use_pi_lines=True, replace_zero_line_B=replacement_B,
                            first_gfl_power_scale=first_gfl_power_scale)
    states = problem.get_state_vars()
    n, m = len(states), len(problem.get_algebraic_vars())
    # Extract the assembled DAE as polynomial terms. State rows are ODE RHSs;
    # algebraic rows contain KCL and device constraints to be eliminated.
    references = list(symbolic_references)
    if symbolic_first_gfl_reference:
        references.insert(0, ("gen 1", "P_ref"))
    reference_vars = []
    for generator_name, parameter_name in dict.fromkeys(references):
        generator = next((g for g in problem.grid.generators if g.name == generator_name), None)
        if generator is None or generator.emt_model is None:
            raise ValueError(f"Unknown EMT generator {generator_name!r}")
        matches = [var for var in generator.emt_model.event_dict
                   if var.name == parameter_name]
        if len(matches) != 1:
            raise ValueError(f"Expected one {parameter_name!r} on {generator_name!r}")
        reference_vars.append(matches[0])
    problem.explicit_input_keys = list(dict.fromkeys(references))
    problem.explicit_input_vars = reference_vars
    S, Phi = problem.build_multilinear_matrices_fast(
        symbolic_inputs=reference_vars or None)
    monomials = [tuple(map(int, S.indices[S.indptr[j]:S.indptr[j + 1]]))
                 for j in range(S.shape[1])]
    A = np.zeros((m, m))
    B = np.zeros((m, n))
    constant = np.zeros(m)
    state_terms = []
    for row in range(n + m):
        for ptr in range(Phi.indptr[row], Phi.indptr[row + 1]):
            term = monomials[Phi.indices[ptr]]
            coeff = float(Phi.data[ptr])
            if row < n:
                state_terms.append((row, term, coeff))
            elif not term:
                constant[row - n] += coeff
            elif len(term) == 1 and term[0] < n:
                B[row - n, term[0]] += coeff
            elif len(term) == 1 and term[0] < n + m:
                A[row - n, term[0] - n] += coeff
            else:
                raise ValueError(f"Nonlinear algebraic term: {term}")

    # Replace each π-line terminal-charge state by q = Q*v. The resulting
    # reduced coordinates are non-charge dynamic states plus 27 bus voltages.
    charge = np.array([i for i, state in enumerate(states)
                       if state.name.startswith(("q_f_", "q_t_"))])
    dynamic = np.array([i for i in range(n) if i not in set(charge)])
    voltage = np.arange(m - 27, m)
    other = np.arange(m - 27)
    charge_rows = []
    for index in charge:
        matches = [row for row in np.flatnonzero(abs(B[:, index]) > 1e-12)
                   if np.count_nonzero(B[row]) == 1 and abs(B[row, index] - 1) < 1e-12]
        if len(matches) != 1:
            raise ValueError(f"Unrecognized charge constraint: {index}")
        charge_rows.append(matches[0])
    remaining = np.array([row for row in range(m) if row not in set(charge_rows)])
    Q = -A[np.ix_(charge_rows, voltage)]
    if np.linalg.matrix_rank(Q) != 27:
        raise ValueError("Bus-voltage replacement needs all 27 charged modes")

    z_count = len(dynamic) + 27
    reference_modes = {n + m + i: z_count + i for i in range(len(reference_vars))}
    z_index = {int(index): position for position, index in enumerate(dynamic)}
    x_poly = [{} for _ in range(n)]
    for index in dynamic:
        x_poly[index] = {(z_index[int(index)],): 1.0}
    for position, index in enumerate(charge):
        x_poly[index] = {(len(dynamic) + j,): float(value)
                         for j, value in enumerate(Q[position]) if value != 0}
    v_poly = [{(len(dynamic) + j,): 1.0} for j in range(27)]

    # qdot has a constant coefficient on each capacitor-current algebraic.
    charge_current = np.zeros((len(charge), len(other)))
    state_only = [{} for _ in range(n)]
    state_y = [[{} for _ in range(m)] for _ in range(n)]
    for row, term, coeff in state_terms:
        x_part = {(): coeff}
        y_index = None
        for index in term:
            if index < n:
                x_part = multiply(x_part, x_poly[index])
            elif index < n + m:
                if y_index is not None:
                    raise ValueError(f"Multiple algebraics in state term: {term}")
                y_index = index - n
            elif index in reference_modes:
                x_part = multiply(x_part, {(reference_modes[index],): 1.0})
            else:
                raise ValueError(f"Unexpected variable in state term: {term}")
        target = state_only[row] if y_index is None else state_y[row][y_index]
        for key, value in x_part.items():
            add(target, key, value)
    for i, row in enumerate(charge):
        for j, col in enumerate(other):
            charge_current[i, j] = state_y[row][col].get((), 0.0)
            if any(key for key in state_y[row][col]):
                raise ValueError("State-dependent charge-current coefficient")
    # This constant matrix solves the remaining algebraic device/network
    # equations and the voltage derivatives after charge substitution.
    matrix = np.block([
        [A[np.ix_(remaining, other)], np.zeros((len(remaining), 27))],
        [charge_current, -Q],
    ])
    if np.linalg.matrix_rank(matrix) != m:
        raise ValueError("Charge-substituted solve is singular")
    lu = linalg.lu_factor(matrix)

    # Assemble one polynomial RHS per solve equation, then solve all monomial
    # coefficient columns at once with the same constant LU factorization.
    rhs = [{} for _ in range(m)]
    for i, row in enumerate(remaining):
        add(rhs[i], (), -constant[row])
        for j in range(n):
            if B[row, j]:
                for key, value in x_poly[j].items():
                    add(rhs[i], key, -B[row, j] * value)
        for j, col in enumerate(voltage):
            if A[row, col]:
                add(rhs[i], (len(dynamic) + j,), -A[row, col])
    for i, row in enumerate(charge):
        target = rhs[len(remaining) + i]
        for key, value in state_only[row].items():
            add(target, key, -value)
        for j, col in enumerate(voltage):
            for key, value in multiply(state_y[row][col], v_poly[j]).items():
                add(target, key, -value)
    keys = sorted(set().union(*(set(poly) for poly in rhs)))
    rhs_matrix = np.array([[poly.get(key, 0.0) for key in keys] for poly in rhs])
    solution = linalg.lu_solve(lu, rhs_matrix)
    y_poly = []
    for row in range(len(other)):
        y_poly.append({key: float(value) for key, value in zip(keys, solution[row])
                       if value != 0})
    y_poly += v_poly

    # Substitute that solve into the differential equations. `output` is the
    # final explicit ODE, still stored as sparse polynomial coefficient maps.
    output = [dict(state_only[row]) for row in dynamic]
    for out_row, source_row in enumerate(dynamic):
        for j in range(m):
            if state_y[source_row][j]:
                for key, value in multiply(state_y[source_row][j], y_poly[j]).items():
                    add(output[out_row], key, value)
    for j in range(27):
        output.append({key: float(value) for key, value in zip(keys, solution[len(other) + j])
                       if value != 0})
    return problem, output, dynamic, charge, Q, matrix


def specialize_reference(output, reference_value):
    """Fix the one input mode (index len(output)) without reducing the grid again."""
    reference_mode = len(output)
    specialized = []
    for row in output:
        fixed_row = {}
        for support, coefficient in row.items():
            if reference_mode in support:
                reduced_support = tuple(index for index in support if index != reference_mode)
                add(fixed_row, reduced_support, coefficient * reference_value)
            else:
                add(fixed_row, support, coefficient)
        specialized.append(fixed_row)
    return specialized


def specialize_inputs(output, values):
    """Fix input modes len(output), len(output)+1, ... at supplied values."""
    specialized = []
    for row in output:
        fixed_row = {}
        for support, coefficient in row.items():
            remaining = []
            value = coefficient
            for mode in support:
                if mode < len(output):
                    remaining.append(mode)
                else:
                    value *= values[mode - len(output)]
            add(fixed_row, tuple(remaining), value)
        specialized.append(fixed_row)
    return specialized


def exact_monomial_cp(output, *, tolerance=1e-10):
    """Return output factors and binary-mode supports for the full ODE.

    A support contains the modes whose factor is [0, 1]; every absent mode
    uses [1, 0]. Thus each column contributes one monomial to all outputs.
    This is an exact construction for retained coefficients, not a claim of
    minimal CP rank.
    """
    supports = sorted({key for row in output for key, value in row.items()
                       if abs(value) > tolerance})
    if any(len(key) != len(set(key)) for key in supports):
        raise ValueError("Repeated state in CP support: ODE is not multi-affine")
    factors = np.array([[row.get(key, 0.0) for key in supports] for row in output])
    return factors, supports


def main():
    problem, output, dynamic, _, _, matrix = extract()
    tolerance = 1e-10
    retained = [(i, key, value) for i, row in enumerate(output)
                for key, value in row.items() if abs(value) > tolerance]
    repeated = [(i, key, value) for i, key, value in retained if len(key) != len(set(key))]
    supports = set(key for _, key, _ in retained)
    print(f"ODE states={len(output)} ({len(dynamic)} non-charge + 27 voltages)")
    print(f"constant solve rank={np.linalg.matrix_rank(matrix)} of {matrix.shape[1]}")
    print(f"retained coefficient entries={len(retained)}, supports={len(supports)}")
    print(f"support degrees={dict(sorted(Counter(map(len, supports)).items()))}")
    print(f"repeated-state monomials={len(repeated)} at tolerance {tolerance:.3e}")
    for row, key, value in sorted(repeated, key=lambda item: -abs(item[2]))[:8]:
        print(f"  equation {row}: {value:.6g} * " + " * ".join(f"z[{j}]" for j in key))
    if not repeated:
        print(f"Exact monomial CP upper bound: {len(supports)} terms")
    # One monomial is one exact binary-mode CP term: the output factor is its
    # 103-vector of coefficients; each input factor selects 1 or z_j.
    output_factors, ordered = exact_monomial_cp(output, tolerance=tolerance)
    direct_rhs, *_ = charge_reduction_data(problem)
    rng = np.random.default_rng(20261013)
    errors = []
    for _ in range(20):
        z = rng.normal(scale=0.1, size=len(output))
        products = np.array([np.prod(z[list(key)]) if key else 1.0 for key in ordered])
        cp_result = output_factors @ products
        reference = direct_rhs(z)
        errors.append(np.linalg.norm(cp_result - reference)
                      / max(np.linalg.norm(reference), 1.0))
    print(f"max CP/direct relative evaluation error={max(errors):.3e}")
    if repeated or max(errors) > 1e-10:
        raise RuntimeError("Multi-affinity or CP verification failed")


if __name__ == "__main__":
    main()
