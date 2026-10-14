"""Check the exact R-L simplification of IEEE-9's zero-charging pi-lines.

For C=0 the terminal equations q=C*v and qdot=i_cap become q=0 and
i_cap=0. Removing those variables is algebraically equivalent to modeling
that terminal as an R-L branch with no shunt. The series-current ODE remains.
"""

import numpy as np

from TensyGridEngine.emt.emt_explicit_converter_demo.experiments.ieee9_gfl_ode_pi_explicit_probe import coefficient_data
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_gfl_ode_reduced import build_problem


def main() -> None:
    problem = build_problem(use_pi_lines=True)
    A, B, _, state_coefficients = coefficient_data(problem)
    states = problem.get_state_vars()
    n, m = B.shape[1], A.shape[1]
    voltage = np.arange(m - 27, m)
    other = np.arange(m - 27)
    charges = np.array([i for i, state in enumerate(states)
                        if state.name.startswith(("q_f_", "q_t_"))])
    charge_rows = []
    for charge in charges:
        rows = [row for row in np.flatnonzero(abs(B[:, charge]) > 1e-12)
                if np.count_nonzero(B[row]) == 1 and abs(B[row, charge] - 1) < 1e-12]
        if len(rows) != 1:
            raise ValueError(f"Charge state {charge} has no unique q=C*v equation")
        charge_rows.append(rows[0])
    charge_rows = np.array(charge_rows)
    Q = -A[np.ix_(charge_rows, voltage)]
    zero_charge = np.flatnonzero(np.all(abs(Q) < 1e-14, axis=1))
    live_charge = np.flatnonzero(np.any(abs(Q) >= 1e-14, axis=1))

    _, Fy = state_coefficients(np.zeros(n))
    zero_cap_current = []
    for charge in charges[zero_charge]:
        columns = np.flatnonzero(abs(Fy[charge]) > 1e-12)
        if len(columns) != 1 or abs(Fy[charge, columns[0]] - 1) > 1e-12:
            raise ValueError(f"Charge derivative {charge} is not a capacitor current")
        zero_cap_current.append(columns[0])
    zero_cap_current = np.array(zero_cap_current)
    if len(np.unique(zero_cap_current)) != len(zero_cap_current):
        raise ValueError("Capacitor-current variable reused by multiple charge states")

    remaining_rows = np.array([row for row in range(m) if row not in set(charge_rows)])
    other_live = np.array([col for col in other if col not in set(zero_cap_current)])
    charged_voltage = np.flatnonzero(np.any(abs(Q[live_charge]) >= 1e-14, axis=0))
    uncharged_voltage = np.array([j for j in range(27) if j not in set(charged_voltage)])

    # KCL and device algebraics, followed by qdot=C*vdot on live terminals.
    # Unknowns are other algebraics, uncapped bus voltages, capped-bus vdot.
    matrix = np.block([
        [A[np.ix_(remaining_rows, other_live)],
         A[np.ix_(remaining_rows, voltage[uncharged_voltage])],
         np.zeros((len(remaining_rows), len(charged_voltage)))],
        [Fy[np.ix_(charges[live_charge], other_live)],
         Fy[np.ix_(charges[live_charge], voltage[uncharged_voltage])],
         -Q[np.ix_(live_charge, charged_voltage)]],
    ])
    rank = np.linalg.matrix_rank(matrix)
    print(f"original pi-line model: {n} states, {m} algebraics")
    print(f"R-L simplification removes {len(zero_charge)} zero-charge states "
          f"and {len(zero_cap_current)} zero capacitor currents")
    print(f"remaining charge states: {len(live_charge)}; "
          f"charged voltage components: {len(charged_voltage)}")
    print(f"direct solve after R-L simplification: {matrix.shape}, "
          f"rank {rank}, nullity {matrix.shape[1] - rank}")


if __name__ == "__main__":
    main()
