#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Compare numerical differences between descriptor matrices A and E computed by:
1) PolynomialMatrixBuilder path (small_signal_tensygrid.py)
2) Native RMS Jacobian path used by SmallSignalStabilityRmsDriver

The script reports global error metrics and the largest entry-wise differences,
including related row/column labels (equation/variable when available).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from PolynomialMatrixBuilder import PolynomialMatrixBuilder
from small_signal_tensygrid import define_problem


@dataclass
class MatrixBundle:
    name: str
    values: np.ndarray
    row_labels: list[str]
    col_labels: list[str]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        if hasattr(value, "value"):
            try:
                return float(value.value)
            except Exception:
                return None
    return None


def _build_v_dict(problem: Any, builder: PolynomialMatrixBuilder) -> dict[str, float]:
    v_dict: dict[str, float] = {}

    try:
        x0 = problem.get_x0()
        for i, sym in enumerate(problem.state_and_algebraic_vars):
            v_dict[str(sym.name)] = float(x0[i])
    except Exception:
        pass

    if hasattr(problem, "_variable_parameters") and hasattr(problem, "_variable_parameters_values"):
        names = problem._variable_parameters
        values = problem._variable_parameters_values
        if names is not None and values is not None:
            for name_obj, val_obj in zip(names, values):
                s_name = str(name_obj.name) if hasattr(name_obj, "name") else str(name_obj)
                val = _safe_float(val_obj)
                if val is not None:
                    v_dict[s_name] = val

    if hasattr(problem, "_constant_parameters") and hasattr(problem, "_constant_params"):
        for n, v in zip(problem._constant_parameters, problem._constant_params):
            s_name = str(n.name) if hasattr(n, "name") else str(n)
            val = _safe_float(v)
            if val is not None:
                v_dict[s_name] = val

    for sym in builder.all_symbols:
        s_name = str(sym.name)
        if s_name not in v_dict:
            if "_invaux" in s_name:
                base = s_name.split("_invaux")[0]
                v_dict[s_name] = (1.0 / v_dict[base]) if base in v_dict and abs(v_dict[base]) > 1e-12 else 1.0
            elif "_mulaux" in s_name:
                base = s_name.split("_mulaux")[0]
                v_dict[s_name] = v_dict[base] if base in v_dict else 1.0
            else:
                v_dict[s_name] = 0.0

    return v_dict


def _to_dense(mat: Any) -> np.ndarray:
    if mat is None:
        return np.empty((0, 0), dtype=float)
    if sp.issparse(mat):
        return mat.toarray()
    if hasattr(mat, "toarray"):
        return mat.toarray()
    return np.asarray(mat)


def _builder_indices(builder: PolynomialMatrixBuilder) -> tuple[list[int], list[int], list[int]]:
    idx_dx: list[int] = []
    idx_vars: list[int] = []
    idx_u: list[int] = []

    for idx, sym in enumerate(builder.all_symbols):
        s_name = sym.name
        if s_name.startswith("dx") or s_name.startswith("xp") or s_name.startswith("d_") or s_name.startswith("dt_"):
            idx_dx.append(idx)
        elif s_name.startswith("u") and not s_name.startswith("u_"):
            idx_u.append(idx)
        else:
            idx_vars.append(idx)

    return idx_dx, idx_vars, idx_u


def _build_polynomial_matrices(
    problem: Any,
    enable_arg_lifting: bool = True,
    enable_trig_lifting: bool = True,
    enable_aux_constraints: bool = True,
) -> tuple[MatrixBundle, MatrixBundle]:
    builder = PolynomialMatrixBuilder(
        problem=problem,
        verbose=False,
        enable_arg_lifting=enable_arg_lifting,
        enable_trig_lifting=enable_trig_lifting,
        enable_aux_constraints=enable_aux_constraints,
    )
    v_dict = _build_v_dict(problem, builder)
    builder.linearize(v_dict=v_dict)

    a = _to_dense(builder.A)
    e = _to_dense(builder.E)

    _, idx_vars, _ = _builder_indices(builder)
    cols_to_take = min(len(idx_vars), a.shape[1]) if a.size else 0
    state_names = [builder.all_symbols[i].name for i in idx_vars[:cols_to_take]]

    row_count = a.shape[0] if a.size else 0
    row_labels = [f"eq[{i}]" for i in range(row_count)]

    if hasattr(builder, "eqs"):
        eqs = builder.eqs
        for i in range(min(len(eqs), row_count)):
            row_labels[i] = str(eqs[i])

    if len(state_names) < a.shape[1]:
        state_names = state_names + [f"col[{i}]" for i in range(len(state_names), a.shape[1])]

    a_bundle = MatrixBundle(
        name="A_poly",
        values=a,
        row_labels=row_labels,
        col_labels=state_names,
    )
    e_bundle = MatrixBundle(
        name="E_poly",
        values=e,
        row_labels=row_labels,
        col_labels=state_names if e.shape[1] == len(state_names) else [f"col[{i}]" for i in range(e.shape[1])],
    )

    return a_bundle, e_bundle


def _build_native_matrices(problem: Any) -> tuple[MatrixBundle, MatrixBundle]:
    x = problem.get_x0()
    dx = np.zeros(problem.get_diff_var_number())

    nx = int(problem.get_states_number())
    ny = int(problem.get_algebraic_var_number())

    h = 1e9
    gy = problem.get_j22(x, dx, h)

    if nx == 0:
        a = _to_dense(gy)
    else:
        fx = problem.get_j11(x, dx, h)
        fy = problem.get_j12(x, dx, h)
        gx = problem.get_j21(x, dx, h)
        a_top = sp.hstack([fx, fy])
        a_bot = sp.hstack([gx, gy])
        a = _to_dense(sp.vstack([a_top, a_bot]))

    e = _to_dense(problem.get_E_matrix(x, dx))

    state_vars = list(getattr(problem, "_state_vars", []))
    alg_vars = list(getattr(problem, "_algebraic_vars", []))
    if nx == 0:
        col_labels = [str(v.name) if hasattr(v, "name") else str(v) for v in alg_vars]
    else:
        col_labels = [str(v.name) if hasattr(v, "name") else str(v) for v in (state_vars + alg_vars)]

    if nx == 0:
        row_eqs = list(getattr(problem, "_algebraic_eqs", []))
    else:
        row_eqs = list(getattr(problem, "_state_eqs", [])) + list(getattr(problem, "_algebraic_eqs", []))
    row_labels = [str(eq) for eq in row_eqs]

    if len(col_labels) < a.shape[1]:
        col_labels = col_labels + [f"col[{i}]" for i in range(len(col_labels), a.shape[1])]
    if len(row_labels) < a.shape[0]:
        row_labels = row_labels + [f"eq[{i}]" for i in range(len(row_labels), a.shape[0])]

    a_bundle = MatrixBundle(
        name="A_native",
        values=a,
        row_labels=row_labels,
        col_labels=col_labels,
    )
    e_bundle = MatrixBundle(
        name="E_native",
        values=e,
        row_labels=row_labels if e.shape[0] == len(row_labels) else [f"eq[{i}]" for i in range(e.shape[0])],
        col_labels=col_labels if e.shape[1] == len(col_labels) else [f"col[{i}]" for i in range(e.shape[1])],
    )

    return a_bundle, e_bundle


def _intersection_indices(left_labels: list[str], right_labels: list[str]) -> tuple[list[int], list[int], list[str]]:
    right_pos = {lbl: j for j, lbl in enumerate(right_labels)}
    li: list[int] = []
    rj: list[int] = []
    common: list[str] = []
    for i, lbl in enumerate(left_labels):
        j = right_pos.get(lbl)
        if j is not None:
            li.append(i)
            rj.append(j)
            common.append(lbl)
    return li, rj, common


def _align_for_compare(
    left: MatrixBundle,
    right: MatrixBundle,
    align_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], str]:
    if align_mode == "common":
        rli, rrj, common_rows = _intersection_indices(left.row_labels, right.row_labels)
        cli, crj, common_cols = _intersection_indices(left.col_labels, right.col_labels)

        if len(common_rows) > 0 and len(common_cols) > 0:
            lmat = left.values[np.ix_(rli, cli)]
            rmat = right.values[np.ix_(rrj, crj)]
            return lmat, rmat, common_rows, common_cols, "common-label"

    if align_mode == "strict":
        if left.values.shape != right.values.shape:
            raise ValueError(f"Shape mismatch in strict mode: {left.values.shape} vs {right.values.shape}")
        rows = left.row_labels if len(left.row_labels) == left.values.shape[0] else [f"row[{i}]" for i in range(left.values.shape[0])]
        cols = left.col_labels if len(left.col_labels) == left.values.shape[1] else [f"col[{i}]" for i in range(left.values.shape[1])]
        return left.values, right.values, rows, cols, "strict-index"

    rows = min(left.values.shape[0], right.values.shape[0])
    cols = min(left.values.shape[1], right.values.shape[1])
    lmat = left.values[:rows, :cols]
    rmat = right.values[:rows, :cols]
    row_labels = [left.row_labels[i] if i < len(left.row_labels) else f"row[{i}]" for i in range(rows)]
    col_labels = [left.col_labels[j] if j < len(left.col_labels) else f"col[{j}]" for j in range(cols)]
    return lmat, rmat, row_labels, col_labels, "truncate-index"


def _relative_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-15)
    return np.abs(a - b) / denom


def compare_matrix(
    left: MatrixBundle,
    right: MatrixBundle,
    atol: float,
    rtol: float,
    top_n: int,
    align_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    a, b, row_labels, col_labels, mode_used = _align_for_compare(left, right, align_mode)

    abs_diff = np.abs(a - b)
    rel_diff = _relative_error(a, b)
    close_mask = np.isclose(a, b, atol=atol, rtol=rtol)

    summary = {
        "shape_left": left.values.shape,
        "shape_right": right.values.shape,
        "shape_compared": a.shape,
        "mode_used": mode_used,
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "max_rel": float(np.max(rel_diff)) if rel_diff.size else 0.0,
        "fro_abs": float(np.linalg.norm(abs_diff, ord="fro")) if abs_diff.size else 0.0,
        "num_fail": int(np.size(close_mask) - np.count_nonzero(close_mask)),
        "num_total": int(np.size(close_mask)),
    }

    if abs_diff.size == 0:
        return [], summary

    flat_idx = np.argsort(abs_diff, axis=None)[::-1]
    rows, cols = np.unravel_index(flat_idx, abs_diff.shape)

    records: list[dict[str, Any]] = []
    for i in range(min(top_n, len(rows))):
        r = int(rows[i])
        c = int(cols[i])
        records.append(
            {
                "row_idx": r,
                "col_idx": c,
                "row_label": row_labels[r] if r < len(row_labels) else f"row[{r}]",
                "col_label": col_labels[c] if c < len(col_labels) else f"col[{c}]",
                "left_val": float(a[r, c]),
                "right_val": float(b[r, c]),
                "abs_diff": float(abs_diff[r, c]),
                "rel_diff": float(rel_diff[r, c]),
                "is_close": bool(close_mask[r, c]),
            }
        )

    return records, summary


def _print_summary(matrix_name: str, summary: dict[str, Any], top: list[dict[str, Any]], show_only_fail: bool) -> None:
    print("\n" + "=" * 88)
    print(f"{matrix_name} comparison")
    print("=" * 88)
    print(f"left shape     : {summary['shape_left']}")
    print(f"right shape    : {summary['shape_right']}")
    print(f"compared shape : {summary['shape_compared']} ({summary['mode_used']})")
    print(f"max |delta|    : {summary['max_abs']:.6e}")
    print(f"max rel delta  : {summary['max_rel']:.6e}")
    print(f"Fro |delta|    : {summary['fro_abs']:.6e}")
    print(f"not close      : {summary['num_fail']} / {summary['num_total']}")

    print("\nTop entry-wise differences:")
    print("idx(row,col) | left | right | abs_diff | rel_diff | close | equation(row) | variable(col)")
    shown = 0
    for rec in top:
        if show_only_fail and rec["is_close"]:
            continue
        print(
            f"({rec['row_idx']},{rec['col_idx']}) | "
            f"{rec['left_val']:.6e} | {rec['right_val']:.6e} | "
            f"{rec['abs_diff']:.6e} | {rec['rel_diff']:.6e} | {rec['is_close']} | "
            f"{rec['row_label']} | {rec['col_label']}"
        )
        shown += 1

    if shown == 0:
        print("(no entries to show with the current filters)")


def _print_order(title: str, row_labels: list[str], col_labels: list[str]) -> None:
    print("\n" + "-" * 88)
    print(title)
    print("-" * 88)
    print(f"rows ({len(row_labels)}):")
    for i, lbl in enumerate(row_labels):
        print(f"  [{i}] {lbl}")
    print(f"cols ({len(col_labels)}):")
    for j, lbl in enumerate(col_labels):
        print(f"  [{j}] {lbl}")


def _write_csv(path: str, matrix_tag: str, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "matrix",
        "row_idx",
        "col_idx",
        "row_label",
        "col_label",
        "left_val",
        "right_val",
        "abs_diff",
        "rel_diff",
        "is_close",
    ]

    mode = "w"
    write_header = True
    if matrix_tag != "A":
        mode = "a"
        write_header = False

    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for rec in rows:
            out = dict(rec)
            out["matrix"] = matrix_tag
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare A/E matrices between polynomial and native small-signal pipelines")
    p.add_argument("--matrix", choices=["A", "E", "both"], default="both", help="Which matrix to compare")
    p.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for np.isclose")
    p.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for np.isclose")
    p.add_argument("--top", type=int, default=25, help="Number of largest differences to print")
    p.add_argument(
        "--align",
        choices=["common", "truncate", "strict"],
        default="common",
        help="Alignment mode: common labels, index truncation, or strict shape match",
    )
    p.add_argument("--only-fail", action="store_true", help="Print only entries failing tolerance")
    p.add_argument("--csv", type=str, default="", help="Optional CSV output path for top differences")
    p.add_argument(
        "--poly-enable-arg-lifting",
        action="store_true",
        help="Enable polynomial argument flattening (argaux). Default: disabled",
    )
    p.add_argument(
        "--poly-enable-trig-lifting",
        action="store_true",
        help="Enable polynomial trig lifting (sin/cos states). Default: disabled",
    )
    p.add_argument(
        "--poly-enable-aux-constraints",
        action="store_true",
        help="Enable polynomial auxiliary constraints (_mulaux/_invaux/_sqrtaux). Default: disabled",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    problem = define_problem()
    if not problem:
        print("Could not create phasor problem.")
        return 1

    a_poly, e_poly = _build_polynomial_matrices(
        problem,
        enable_arg_lifting=args.poly_enable_arg_lifting,
        enable_trig_lifting=args.poly_enable_trig_lifting,
        enable_aux_constraints=args.poly_enable_aux_constraints,
    )
    a_native, e_native = _build_native_matrices(problem)

    _print_order("A_poly order (left)", a_poly.row_labels, a_poly.col_labels)
    _print_order("A_native order (right)", a_native.row_labels, a_native.col_labels)
    _print_order("E_poly order (left)", e_poly.row_labels, e_poly.col_labels)
    _print_order("E_native order (right)", e_native.row_labels, e_native.col_labels)

    if args.matrix in ("A", "both"):
        a_rows, a_summary = compare_matrix(
            left=a_poly,
            right=a_native,
            atol=args.atol,
            rtol=args.rtol,
            top_n=args.top,
            align_mode=args.align,
        )
        _print_summary("A (poly vs native)", a_summary, a_rows, show_only_fail=args.only_fail)
        if args.csv:
            _write_csv(args.csv, "A", a_rows)

    if args.matrix in ("E", "both"):
        e_rows, e_summary = compare_matrix(
            left=e_poly,
            right=e_native,
            atol=args.atol,
            rtol=args.rtol,
            top_n=args.top,
            align_mode=args.align,
        )
        _print_summary("E (poly vs native)", e_summary, e_rows, show_only_fail=args.only_fail)
        if args.csv:
            _write_csv(args.csv, "E", e_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
