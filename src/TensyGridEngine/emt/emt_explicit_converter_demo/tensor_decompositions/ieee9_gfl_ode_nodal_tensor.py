"""Measure bus-phase separation rank of the reduced IEEE-9 nodal inverse.

The 102 assembled algebraic equations are partitioned into 75 local device
equations followed by 27 bus KCL equations. Eliminating the local currents
gives a 27-by-27 nodal matrix. Its inverse is tested as a sum of bus-matrix
outer products with phase-matrix factors.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from TensyGridEngine.emt.emt_explicit_converter_demo.ieee9_gfl_ode_reduced import build_problem
from VeraGridEngine.Utils.Symbolic.symbolic import Const


def nodal_matrix(problem) -> np.ndarray:
    """Eliminate local algebraic currents, retaining bus phase voltages."""
    variables = problem.get_algebraic_vars()
    equations = problem._algebraic_eqs
    n_voltage = 3 * len(problem.grid.buses)
    n_local = len(variables) - n_voltage
    bus_uids = {variable.uid for bus in problem.grid.buses
                for variable in bus.emt_model.algebraic_vars}
    if {variable.uid for variable in variables[n_local:]} != bus_uids:
        raise ValueError("The final algebraic variables are not the bus voltages")
    devices = [*problem.grid.lines, *problem.grid.generators, *problem.grid.loads]
    if sum(len(device.emt_model.algebraic_eqs) for device in devices) != n_local:
        raise ValueError("Local equations do not precede the bus KCL equations")

    parameter_values = {
        variable: Const(float(value))
        for variable, value in zip(problem._variable_parameters, problem._event_params_values)
    }
    parameter_values.update({
        variable: Const(float(value))
        for variable, value in zip(problem._constant_parameters, problem._constant_params_values)
    })
    column_by_uid = {variable.uid: index for index, variable in enumerate(variables)}
    matrix = np.zeros((len(equations), len(variables)))
    for row, equation in enumerate(equations):
        involved = {variable.uid: variable for variable in equation.get_vars()
                    if variable.uid in column_by_uid}
        for variable in involved.values():
            coefficient = equation.diff(variable).subs(parameter_values).simplify()
            if not isinstance(coefficient, Const):
                raise ValueError(f"Equation {row} is not constant-linear in {variable.name}")
            matrix[row, column_by_uid[variable.uid]] = float(coefficient.value)

    local = matrix[:n_local, :n_local]
    if np.linalg.matrix_rank(local) != n_local:
        raise ValueError("Local device equations cannot be eliminated")
    # Schur complement after substituting the local device currents into KCL.
    return matrix[n_local:, n_local:] - matrix[n_local:, :n_local] @ linalg.solve(
        local, matrix[:n_local, n_local:]
    )


def main() -> None:
    problem = build_problem()
    conductance = nodal_matrix(problem)
    n_bus = len(problem.grid.buses)
    inverse = linalg.inv(conductance)

    # Group row/column indices as (bus, phase, bus', phase'). An SVD of the
    # (bus, bus') versus (phase, phase') unfolding gives the best rank-r sum
    # of bus-matrix ⊗ phase-matrix terms.
    tensor = inverse.reshape(n_bus, 3, n_bus, 3).transpose(0, 2, 1, 3)
    unfolding = tensor.reshape(n_bus * n_bus, 9)
    left, singular_values, right = linalg.svd(unfolding, full_matrices=False)
    rank_one = np.outer(left[:, 0] * singular_values[0], right[0])
    rank_two = (left[:, :2] * singular_values[:2]) @ right[:2]
    rank_one_error = np.linalg.norm(unfolding - rank_one) / np.linalg.norm(unfolding)
    rank_two_error = np.linalg.norm(unfolding - rank_two) / np.linalg.norm(unfolding)

    common = np.ones((3, 3)) / 3.0
    differential = np.eye(3) - common
    bus_common = np.einsum("ijpq,pq->ij", tensor, common)
    bus_differential = np.einsum("ijpq,pq->ij", tensor, differential) / 2.0
    projector_form = (
        np.einsum("ij,pq->ijpq", bus_common, common)
        + np.einsum("ij,pq->ijpq", bus_differential, differential)
    )
    projector_error = np.linalg.norm(tensor - projector_form) / np.linalg.norm(tensor)

    rng = np.random.default_rng(20261008)
    rhs = rng.standard_normal((n_bus, 3))
    direct = linalg.solve(conductance, rhs.ravel()).reshape(n_bus, 3)
    phase_mean = rhs.mean(axis=1, keepdims=True)
    factored = bus_common @ phase_mean + bus_differential @ (rhs - phase_mean)
    solve_error = np.linalg.norm(direct - factored) / np.linalg.norm(direct)
    if rank_two_error > 1e-10 or projector_error > 1e-10 or solve_error > 1e-10:
        raise RuntimeError("Two-factor nodal inverse failed reconstruction checks")

    print(f"nodal matrix={conductance.shape}, condition={np.linalg.cond(conductance):.3e}")
    print(f"singular values={singular_values}")
    print(f"rank-1 relative error={rank_one_error:.3e}")
    print(f"rank-2 relative error={rank_two_error:.3e}")
    print(f"common/differential projector error={projector_error:.3e}")
    print(f"factored solve error={solve_error:.3e}")


if __name__ == "__main__":
    main()
