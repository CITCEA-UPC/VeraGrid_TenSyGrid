# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sys
from pathlib import Path

import scipy
import numpy as np

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from TensyGridEngine.cpn_utils import build_problems, process_cpn_system


def run_small_signal_from_driver(problem, pf_results, rms_options):
    import VeraGridEngine.api as vge
    ss_options = vge.RmsSmallSignalStabilityOptions(ss_assessment_time=0, verbose=0)
    driver = vge.SmallSignalStabilityRmsDriver(
        grid=vge.MultiCircuit(Sbase=problem.grid.Sbase),
        rms_options=rms_options,
        sss_options=ss_options,
        pf_results=pf_results,
    )
    driver.problem = problem
    driver.k = problem.get_states_number()
    driver.run()
    state_var_names = [str(v.name) if hasattr(v, 'name') else f"state_{i}"
                       for i, v in enumerate(problem.state_and_algebraic_vars)]
    return driver.results.eigenvalues, driver.results.participation_factors, state_var_names


def main() -> None:
    """Export the processed CPN1 matrices to a MATLAB file."""
    JAUME_FLAG = True
    grid_filename = sys.argv[1] if len(sys.argv) > 1 else "IEEE 9 Bus.gridcal"
    problem_ml, pf_results, rms_options_ml = build_problems(grid_filename=grid_filename)

    problem_ml.build_multilinear_matrices()

    print(f"[INFO] Original S: {problem_ml.S.shape}, Phi: {problem_ml.Phi.shape}")
    print(f"[INFO] Parameters: {len(problem_ml._constant_parameters)} constant, {len(problem_ml._variable_parameters)} variable")

    cpn = process_cpn_system(problem_ml, jaume_flag=JAUME_FLAG)

    n_eqs = cpn.Phi.shape[0]
    n_vars = cpn.S.shape[0]
    n_mon = cpn.S.shape[1]
    rank_Phi = np.linalg.matrix_rank(cpn.Phi.toarray())

    print(f"[RESULT] Equations: {n_eqs}, Variables: {n_vars}, Monomials: {n_mon}, Rank(Phi): {rank_Phi}")
    print(f"[RESULT] System is {'well-determined' if rank_Phi == n_vars else 'UNDERDETERMINED (rank < n_vars)'}")

    eqs_array = np.array(cpn.eqs_list, dtype=object)
    vars_array = np.array(cpn.vars_list, dtype=object)
    output_dir: Path = Path(__file__).resolve().parent / "CPN1_computations"
    output_dir.mkdir(exist_ok=True)
    output_path: Path = output_dir / f"CPN1_{problem_ml.grid}.mat"
    scipy.io.savemat(str(output_path), {
        'S': cpn.S,
        'Phi': cpn.Phi,
        'eqs': eqs_array,
        'var': vars_array
    })

    print(f"CPN1 file saved: {output_path}")


if __name__ == "__main__":
    main()
