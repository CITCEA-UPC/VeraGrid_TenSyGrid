# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from scipy import sparse

from TensyGridEngine.emt.emt_ieee9 import (
    attach_emt_models,
    build_emt_options,
    build_ieee9_grid,
    build_power_flow_options,
)

from VeraGridEngine.Simulations.EMT.problems.emt_problem_multilinear import EmtProblemMultilinear
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver


def build_multilinear_problem_and_matrices() -> tuple[EmtProblemMultilinear, sparse.csr_matrix, sparse.csc_matrix]:
    """Build the IEEE9 multilinear EMT problem and return ``(problem, Phi, S)``."""
    grid = build_ieee9_grid()
    attach_emt_models(grid)

    balanced_pf_driver = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    balanced_pf_driver.run()
    pf_driver = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf_driver.run()
    if not bool(pf_driver.results.converged):
        raise RuntimeError("Three-phase IEEE 9 power flow did not converge")

    problem = EmtProblemMultilinear(
        grid=grid,
        options=build_emt_options(),
        pf_results_3ph=pf_driver.results,
        pf_results=balanced_pf_driver.results,
    )
    Phi, S = problem.linearize_matrices()
    return problem, Phi, S


def main() -> None:
    problem, Phi, S = build_multilinear_problem_and_matrices()
    print("IEEE 9 multilinear EMT matrices")
    print(f"states={len(problem._state_vars)}, algebraic={len(problem._algebraic_vars)}, diff={len(problem._diff_vars)}")
    print(f"Phi shape={Phi.shape}, nnz={Phi.nnz}")
    print(f"S shape={S.shape}, nnz={S.nnz}")


if __name__ == "__main__":
    main()
