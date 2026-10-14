"""Explicit IEEE-9 pi-line ODE after substituting terminal charge q = C*v.

This controlled experiment replaces the three zero line-charging B values by
the specified positive value. It retains all pi-line series-current states,
replaces the 54 terminal-charge states by 27 bus-voltage states, and eliminates
the remaining algebraic currents through one constant linear solve. No KCL
equation is differentiated. This is a modified-capacitance grid, not the
original IEEE-9 charging data.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp

from TensyGridEngine.emt.emt_explicit_converter_demo.experiments.ieee9_gfl_ode_pi_explicit_probe import coefficient_data
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_gfl_ode_reduced import build_problem
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver


def charge_reduction_data(problem):
    """Build q=Q*v and the constant solve for other algebraics and v-dot."""
    A, B, constant, state_coefficients = coefficient_data(problem)
    states = problem.get_state_vars()
    n = len(states)
    m = len(problem.get_algebraic_vars())
    voltage_indices = np.arange(m - 27, m)
    other_indices = np.arange(m - 27)
    charge_indices = np.array([
        index for index, state in enumerate(states)
        if state.name.startswith(("q_f_", "q_t_"))
    ])
    if len(charge_indices) != 54:
        raise ValueError("Expected 54 pi-line terminal-charge states")
    charge_rows = []
    for charge_index in charge_indices:
        matches = [
            row for row in np.flatnonzero(abs(B[:, charge_index]) > 1e-12)
            if np.count_nonzero(B[row]) == 1 and abs(B[row, charge_index] - 1.0) < 1e-12
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot isolate charge constraint for state {charge_index}")
        charge_rows.append(matches[0])
    charge_rows = np.array(charge_rows)
    remaining_rows = np.array([row for row in range(m) if row not in set(charge_rows)])
    Q = -A[np.ix_(charge_rows, voltage_indices)]
    if (np.linalg.norm(A[np.ix_(charge_rows, other_indices)]) > 1e-12
            or np.linalg.norm(constant[charge_rows]) > 1e-12):
        raise ValueError("Charge constraints are not q=Q*v")
    bus_capacitance = np.zeros((27, 27))
    for row, charge_index in enumerate(charge_indices):
        connected = np.flatnonzero(abs(Q[row]) > 1e-14)
        buses = set(connected // 3)
        if len(buses) != 1:
            raise ValueError("A terminal charge must belong to one bus")
        bus = buses.pop()
        phase = "ABC".index(states[charge_index].name[-1])
        bus_capacitance[3 * bus + phase] += Q[row]
    if np.linalg.matrix_rank(bus_capacitance) != 27:
        raise ValueError("The aggregate bus capacitance matrix is singular")

    x_zero = np.zeros(n)
    _, Fy_zero = state_coefficients(x_zero)
    charge_current_matrix = Fy_zero[np.ix_(charge_indices, other_indices)]
    matrix = np.block([
        [A[np.ix_(remaining_rows, other_indices)], np.zeros((len(remaining_rows), 27))],
        [charge_current_matrix, -Q],
    ])
    if np.linalg.matrix_rank(matrix) != m:
        raise ValueError("Charge-substituted network still has unsolved algebraics")
    lu = linalg.lu_factor(matrix)
    dynamic_indices = np.array([index for index in range(n) if index not in set(charge_indices)])

    def vector_field(z):
        dynamic = z[:len(dynamic_indices)]
        voltage = z[len(dynamic_indices):]
        x = np.zeros(n)
        x[dynamic_indices] = dynamic
        x[charge_indices] = Q @ voltage
        f0, Fy = state_coefficients(x)
        if np.linalg.norm(Fy[np.ix_(charge_indices, other_indices)] - charge_current_matrix) > 1e-10:
            raise RuntimeError("Charge-current coefficient matrix became state-dependent")
        rhs = np.r_[
            -(B[remaining_rows] @ x + A[np.ix_(remaining_rows, voltage_indices)] @ voltage
              + constant[remaining_rows]),
            -(f0[charge_indices] + Fy[np.ix_(charge_indices, voltage_indices)] @ voltage),
        ]
        solution = linalg.lu_solve(lu, rhs)
        other_algebraics = solution[:len(other_indices)]
        voltage_derivative = solution[len(other_indices):]
        y = np.r_[other_algebraics, voltage]
        state_derivative = f0 + Fy @ y
        charge_residual = np.linalg.norm(state_derivative[charge_indices] - Q @ voltage_derivative)
        charge_scale = max(np.linalg.norm(state_derivative[charge_indices]), 1.0)
        if charge_residual / charge_scale > 1e-9:
            raise RuntimeError(f"Charge derivative residual is {charge_residual:.3e}")
        return np.r_[state_derivative[dynamic_indices], voltage_derivative]

    return vector_field, dynamic_indices, charge_indices, Q, matrix, bus_capacitance


def main(*, replacement_B: float = 0.1, step: float = 2.5e-6) -> None:
    full = build_problem(use_pi_lines=True, replace_zero_line_B=replacement_B)
    n = len(full.get_state_vars())
    m = len(full.get_algebraic_vars())
    full.options.time_step = step
    solver = build_emt_solver(
        options=full.options, problem=full, t0=0.0, t_end=1e-4,
        h=step, method=full.options.integration_method,
    )
    times, values, _, initialized, converged = solver.simulate(boundary_updater=full)
    if not (initialized and converged):
        raise RuntimeError("Full charged pi-line DAE simulation failed")

    vector_field, dynamic_indices, charge_indices, Q, matrix, bus_capacitance = charge_reduction_data(full)
    start_sample = 1  # The reported t=0 point precedes DAE constraint projection.
    voltage_start = values[start_sample, n + m - 27:n + m]
    z0 = np.r_[values[start_sample, dynamic_indices], voltage_start]
    initial_charge_error = np.linalg.norm(values[start_sample, charge_indices] - Q @ voltage_start)
    result = solve_ivp(lambda _, z: vector_field(z),
                       (times[start_sample], times[-1]), z0,
                       t_eval=times[start_sample:], rtol=1e-9, atol=1e-11)
    if not result.success:
        raise RuntimeError(result.message)
    reference = np.column_stack((
        values[start_sample:, dynamic_indices],
        values[start_sample:, n + m - 27:n + m],
    ))
    error = np.max(np.abs(result.y.T - reference))
    scale = max(np.max(np.abs(reference)), 1.0)
    print(f"replacement zero-line B={replacement_B:g}, DAE step={step:g} s")
    print(f"full DAE: states={n}, algebraics={m}; explicit ODE states={len(z0)}")
    print(f"aggregate bus capacitance rank={np.linalg.matrix_rank(bus_capacitance)} of 27")
    print(f"charge-substituted solve rank={np.linalg.matrix_rank(matrix)} of {m}")
    print(f"initial charge constraint error={initial_charge_error:.3e}")
    print(f"max trajectory difference={error:.3e} (relative={error / scale:.3e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacement-B", type=float, default=0.1)
    parser.add_argument("--step", type=float, default=2.5e-6)
    args = parser.parse_args()
    main(replacement_B=args.replacement_B, step=args.step)
