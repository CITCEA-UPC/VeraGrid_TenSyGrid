# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Export the CPN1 multilinear system (S, Phi, equations, variables) to .mat.

Configuration flags (set in main() call, see __name__ == "__main__" block):

    SIMPLIFY : bool (default True)
        Applies the full simplification pipeline:
          - merge_duplicate_monomials (sums Phi columns with identical S vectors)
          - fixpoint loop: regenerate_eqs_list -> simplify_system -> substitute_pinned_constants
          - handle_free_variables -> reduce_rank_qr -> pin_vars_without_trivial_monomials -> cleanup
        When False, returns the raw system after parameter substitution.

    ZERO_DERIVATIVES : bool (default True)
        Removes derivative columns (dx/dt -> 0 substitution).
        When False, keeps derivative variables in the system.

    EXPORT_EXCEL : bool (default True)
        Writes a .xlsx workbook with sheets for Summary, Variables, Equations,
        S matrix (COO), Phi matrix (COO), and Jacobian evaluated at x0.

Usage:
    python export_CPN_data.py <grid_filename>
"""

# ── Imports ──

from __future__ import annotations

import sys
from pathlib import Path

import scipy
import numpy as np

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from TensyGridEngine.cpn_utils import build_problems, process_cpn_system, compute_jacobian_from_S


# ── Helpers ──

def _export_excel(cpn, problem, x0, n_eqs, n_vars, n_mon, rank_Phi, F, J, grid_stem=""):
    """Write S, Phi, equations, variables, and Jacobian to an .xlsx workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    out_dir = Path('CPN1_computations')
    out_dir.mkdir(exist_ok=True)
    stem = grid_stem or str(problem.grid)
    filepath = out_dir / f'{stem}_stats.xlsx'

    wb = Workbook()
    hdr_font = Font(bold=True, size=11)
    hdr_fill = PatternFill(start_color='D9E1F2', end_color='D9E1F2', fill_type='solid')
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin'),
    )

    def _write_header(ws, headers, row=1):
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=row, column=c, value=h)
            cell.font = hdr_font
            cell.fill = hdr_fill
            cell.border = thin_border
            cell.alignment = Alignment(horizontal='center')

    def _auto_width(ws):
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    ws = wb.active
    ws.title = 'Summary'
    rows = [
        ('Grid', str(problem.grid)),
        ('n_eqs', n_eqs),
        ('n_vars', n_vars),
        ('n_mon', n_mon),
        ('Rank(Phi)', rank_Phi),
        ('Well-determined', rank_Phi == n_vars),
    ]
    _write_header(ws, ['Property', 'Value'])
    for r, (k, v) in enumerate(rows, 2):
        ws.cell(row=r, column=1, value=k).font = Font(bold=True)
        ws.cell(row=r, column=2, value=v)
    _auto_width(ws)

    ws = wb.create_sheet('Variables')
    _write_header(ws, ['idx', 'name', 'x0'])
    for i, var in enumerate(cpn.vars_list):
        ws.cell(row=i + 2, column=1, value=i)
        ws.cell(row=i + 2, column=2, value=var)
        ws.cell(row=i + 2, column=3, value=float(x0[i]) if i < len(x0) else None)
    _auto_width(ws)

    ws = wb.create_sheet('Equations')
    _write_header(ws, ['idx', 'equation'])
    for i, eq in enumerate(cpn.eqs_list):
        ws.cell(row=i + 2, column=1, value=i)
        ws.cell(row=i + 2, column=2, value=str(eq))
    _auto_width(ws)

    ws = wb.create_sheet('S_coo')
    S_coo = cpn.S.tocoo()
    _write_header(ws, ['row(var)', 'col(mon)', 'value'])
    for r, c, v in zip(S_coo.row, S_coo.col, S_coo.data):
        row_num = ws.max_row + 1
        ws.cell(row=row_num, column=1, value=int(r))
        ws.cell(row=row_num, column=2, value=int(c))
        ws.cell(row=row_num, column=3, value=float(v))
    _auto_width(ws)

    ws = wb.create_sheet('Phi_coo')
    Phi_coo = cpn.Phi.tocoo()
    _write_header(ws, ['row(eq)', 'col(mon)', 'value'])
    for r, c, v in zip(Phi_coo.row, Phi_coo.col, Phi_coo.data):
        row_num = ws.max_row + 1
        ws.cell(row=row_num, column=1, value=int(r))
        ws.cell(row=row_num, column=2, value=int(c))
        ws.cell(row=row_num, column=3, value=float(v))
    _auto_width(ws)

    ws = wb.create_sheet('Jacobian')
    J_dense = J.toarray() if hasattr(J, 'toarray') else np.asarray(J)
    _write_header(ws, [''] + [f'v{j}' for j in range(J_dense.shape[1])])
    for i in range(J_dense.shape[0]):
        ws.cell(row=i + 2, column=1, value=f'eq{i}').font = Font(bold=True)
        for j in range(J_dense.shape[1]):
            val = J_dense[i, j]
            if abs(val) > 1e-15:
                ws.cell(row=i + 2, column=j + 2, value=round(val, 6))
    _auto_width(ws)

    wb.save(filepath)
    print(f"Excel saved: {filepath}")


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


# ── Main entry ──

def main(grid_file: str = "red_enana.gridcal",
         SIMPLIFY: bool = True,
         ZERO_DERIVATIVES: bool = True,
         EXPORT_EXCEL: bool = True) -> None:
    """Build the CPN1 steady-state system and export to .mat.

    Parameters
    ----------
    grid_file : str, default "red_enana.gridcal"
        Grid file name to load.
    SIMPLIFY : bool, default True
        Apply simplification pipeline (deduplicate, substitute pinned variables).
    ZERO_DERIVATIVES : bool, default True
        Remove derivative columns (dx/dt -> 0).
    EXPORT_EXCEL : bool, default True
        Export an .xlsx workbook alongside the .mat file.
    """
    grid_filename = sys.argv[1] if len(sys.argv) > 1 else grid_file
    problem_ml, pf_results, rms_options_ml = build_problems(grid_filename=grid_filename)

    problem_ml.build_multilinear_matrices()

    print(f"[INFO] Original S: {problem_ml.S.shape}, Phi: {problem_ml.Phi.shape}")
    print(f"[INFO] Parameters: {len(problem_ml._constant_parameters)} constant, {len(problem_ml._variable_parameters)} variable")

    cpn = process_cpn_system(problem_ml, jaume_flag=SIMPLIFY, zero_derivatives=ZERO_DERIVATIVES)

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

    if EXPORT_EXCEL:
        x0 = cpn.x0[cpn.orig_keep_idx]
        F = compute_jacobian_from_S(cpn.S, x0)
        J = cpn.Phi @ F.T
        _export_excel(
            cpn=cpn, problem=problem_ml, x0=x0,
            grid_stem=Path(grid_filename).stem,
            n_eqs=n_eqs, n_vars=n_vars, n_mon=n_mon,
            rank_Phi=rank_Phi, F=F, J=J,
        )


if __name__ == "__main__":
    # Entry point: export CPN1 system for the given grid with default flags
    main("rlc_serie.gridcal", True, True, True)
