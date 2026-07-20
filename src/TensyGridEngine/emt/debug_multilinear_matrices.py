# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Debug script to validate S/Phi construction for multilinear RMS.

It compares:
1) RmsProblemMultilinear.build_multilinear_matrices()
2) PolynomialMatrixBuilder internal S_H / Phi_H
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import VeraGridEngine.api as vge
from VeraGridEngine.Utils.Symbolic.compiled_functions import SymbolicJacobian
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from VeraGridEngine.Utils.Symbolic.symbolic import get_expr_factors
from PolynomialMatrixBuilder import PolynomialMatrixBuilder


def create_problem(hard_sat_type: str = "ml") -> vge.RmsProblemMultilinear:
    sbase = 100.0
    grid = vge.MultiCircuit(Sbase=sbase, fbase=50.0)

    bus0 = vge.Bus(name="Bus0", Vnom=10, is_slack=True)
    bus1 = vge.Bus(name="Bus1", Vnom=10)
    grid.add_bus(bus0)
    grid.add_bus(bus1)

    for bus in grid.buses:
        vge.initialize_bus_phasor_rms(bus, vf=grid.var_factory)

    line = vge.Line(
        name="Line",
        bus_from=bus0,
        bus_to=bus1,
        r=0.029585798816568046,
        x=0.07100591715976332,
        b=0.03,
        rate=900.0,
    )
    grid.add_line(line)

    load = vge.Load(P=9.999999, Q=0.999999)
    grid.add_load(bus=bus1, api_obj=load)

    gen = vge.Generator(name="Gen0", P=10, vset=1.0, Snom=900)
    grid.add_generator(bus=bus0, api_obj=gen)

    genqec = vge.get_complete_generator_template_phasor(
        grid.var_factory,
        name="Gen0",
        hard_sat_type=hard_sat_type,
    ).block
    genqec = vge.to_implicit(genqec, grid.var_factory)

    line_mdl = vge.get_line_phasor_rms_template(grid.var_factory).block
    grid.var_factory.add_connections([line_mdl.in_vars[0]], [bus0.rms_model.out_vars[0]])
    grid.var_factory.add_connections([line_mdl.in_vars[1]], [bus0.rms_model.out_vars[1]])
    grid.var_factory.add_connections([line_mdl.in_vars[2]], [bus1.rms_model.out_vars[0]])
    grid.var_factory.add_connections([line_mdl.in_vars[3]], [bus1.rms_model.out_vars[1]])
    set_rms_model(device=line, model=line_mdl, var_factory=grid.var_factory)

    grid.var_factory.add_connections([genqec.in_vars[0]], [bus0.rms_model.out_vars[0]])
    grid.var_factory.add_connections([genqec.in_vars[1]], [bus0.rms_model.out_vars[1]])
    set_rms_model(device=gen, model=genqec, var_factory=grid.var_factory)

    load_mdl = vge.get_load_phasor_current_rms_template(grid.var_factory).block
    grid.var_factory.add_connections([load_mdl.in_vars[0]], [bus1.rms_model.out_vars[0]])
    grid.var_factory.add_connections([load_mdl.in_vars[1]], [bus1.rms_model.out_vars[1]])
    set_rms_model(device=load, model=load_mdl, var_factory=grid.var_factory)

    pf_results = vge.power_flow(grid, vge.PowerFlowOptions())
    if not pf_results.converged:
        raise RuntimeError("Power flow failed")

    rms_options = vge.RmsOptions(
        time_step=0.01,
        simulation_time=1.0,
        tolerance=1e-6,
        max_iter=20,
        problem_type=vge.RmsProblemTypes.Multilinear,
    )
    return vge.RmsProblemMultilinear(grid=grid, options=rms_options, pf_results=pf_results)


def matrix_fingerprint(m: sparse.spmatrix) -> tuple[int, int, int, float]:
    csr = m.tocsr()
    return csr.shape[0], csr.shape[1], csr.nnz, float(np.sum(np.abs(csr.data)))


def monomial_signature_from_S(s_mat: sparse.spmatrix, col_idx: int) -> str:
    csc = s_mat.tocsc()
    start = csc.indptr[col_idx]
    end = csc.indptr[col_idx + 1]
    parts = []
    for p in range(start, end):
        r = int(csc.indices[p])
        v = float(csc.data[p])
        parts.append(f"({r}:{v:+.3f})")
    if len(parts) == 0:
        return "<const>"
    return " * ".join(parts)


def print_monomial_samples(name: str,
                           s_mat: sparse.spmatrix,
                           phi_mat: sparse.spmatrix,
                           max_eq: int = 4,
                           max_terms_per_eq: int = 6) -> None:
    print(f"\n[{name}] monomial separation samples")
    phi_csr = phi_mat.tocsr()

    n_eq = min(max_eq, phi_csr.shape[0])
    for eq_idx in range(n_eq):
        start = phi_csr.indptr[eq_idx]
        end = phi_csr.indptr[eq_idx + 1]
        cols = phi_csr.indices[start:end]
        vals = phi_csr.data[start:end]

        print(f"  eq[{eq_idx}] has {len(cols)} monomial terms")
        for j, (col_idx, coeff) in enumerate(zip(cols, vals)):
            if j >= max_terms_per_eq:
                print("    ...")
                break
            sig = monomial_signature_from_S(s_mat, int(col_idx))
            print(f"    phi={float(coeff):+.6e} | mon[{int(col_idx)}] = {sig}")


def _build_param_subs(problem) -> dict:
    subs_map = {}
    for p, val in zip(problem._variable_parameters, problem._variable_parameters_values):
        subs_map[p] = vge.Const(float(val))
    for p, val in zip(problem._constant_parameters, problem._constant_params):
        subs_map[p] = vge.Const(float(val))
    return subs_map


def print_rejected_terms(problem) -> None:
    all_eqs = list(problem._state_eqs) + list(problem._algebraic_eqs)
    all_vars = list(problem._state_vars) + list(problem._algebraic_vars) + list(problem._diff_vars)
    uid_to_idx = {v.uid: i for i, v in enumerate(all_vars)}

    param_by_uid = {}
    for p, val in zip(problem._variable_parameters, problem._variable_parameters_values):
        param_by_uid[p.uid] = float(val)
    for p, val in zip(problem._constant_parameters, problem._constant_params):
        param_by_uid[p.uid] = float(val)

    all_eqs = [eq.subs(_build_param_subs(problem)).simplify() for eq in all_eqs]

    rejected = []
    total_mono = 0
    for eq_idx, eq in enumerate(all_eqs):
        terms = problem._collect_monomials(eq)
        for mono in terms:
            total_mono += 1
            monom_tuple, _gain = problem._term_to_monomial(
                factors=get_expr_factors(mono),
                base_gain=1.0,
                uid_to_idx=uid_to_idx,
                param_by_uid=param_by_uid,
            )
            if monom_tuple is None:
                rejected.append((eq_idx, str(mono)))

    print("\n[Monomial Parse]")
    print(f"  total candidate monomials: {total_mono}")
    print(f"  rejected monomials: {len(rejected)}")
    for i, (eq_idx, txt) in enumerate(rejected[:12]):
        print(f"    eq[{eq_idx}] rejected: {txt}")
    if len(rejected) > 12:
        print("    ...")

    if len(rejected) > 0:
        print("\n[Rejected Term Deep Dive]")
        for eq_idx, txt in rejected[:2]:
            print(f"  eq[{eq_idx}] raw: {txt}")
            eq_expr = all_eqs[eq_idx]
            mono_terms = problem._collect_monomials(eq_expr)
            print(f"    collected monomials: {len(mono_terms)}")
            for k, mono in enumerate(mono_terms[:12]):
                mono_txt = str(mono)
                factors = get_expr_factors(mono)
                print(f"    mono[{k}]: {mono_txt}")
                print(f"      factors={len(factors)}")
                for fi, f in enumerate(factors):
                    vars_f = f.get_vars()
                    var_names = [str(v.name) for v in vars_f]
                    print(f"        f[{fi}]={f} | n_vars={len(vars_f)} | vars={var_names}")


def _compute_monomial_values(s_mat: sparse.spmatrix, v: np.ndarray) -> np.ndarray:
    s_csc = s_mat.tocsc()
    n_mon = s_csc.shape[1]
    out = np.ones(n_mon, dtype=float)
    for j in range(n_mon):
        start = s_csc.indptr[j]
        end = s_csc.indptr[j + 1]
        prod = 1.0
        for p in range(start, end):
            i = int(s_csc.indices[p])
            s_ij = float(s_csc.data[p])
            prod *= (s_ij * v[i] + (1.0 - abs(s_ij)))
        out[j] = prod
    return out


def print_residual_reconstruction(problem, s_mat: sparse.spmatrix, phi_mat: sparse.spmatrix) -> None:
    x = problem.get_x0()
    n_state_alg = len(problem._state_vars) + len(problem._algebraic_vars)
    n_diff = len(problem._diff_vars)
    dx = np.zeros(n_diff, dtype=float)

    if len(x) != n_state_alg:
        raise RuntimeError(f"x size mismatch: len(x)={len(x)}, n_state_alg={n_state_alg}")

    v = np.zeros(n_state_alg + n_diff, dtype=float)
    v[:n_state_alg] = x

    mon_vals = _compute_monomial_values(s_mat, v)
    r_ml = (phi_mat @ mon_vals).astype(float)

    if len(problem._state_eqs) > 0:
        r_state = problem.rhs_state(x, dx)
    else:
        r_state = np.zeros(0, dtype=float)
    r_alg = problem.rhs_algebraic(x, dx)
    r_sym = np.concatenate([r_state, r_alg])

    n = min(len(r_ml), len(r_sym))
    err = r_ml[:n] - r_sym[:n]
    print("\n[Residual Reconstruction]")
    print(f"  size ml={len(r_ml)} sym={len(r_sym)} compare_n={n}")
    print(f"  max|ml-sym|={np.max(np.abs(err)):.6e}")
    print(f"  mean|ml-sym|={np.mean(np.abs(err)):.6e}")


def print_jacobian_reconstruction(problem, s_mat: sparse.spmatrix, phi_mat: sparse.spmatrix) -> None:
    x = problem.get_x0()
    n_state_alg = len(problem._state_vars) + len(problem._algebraic_vars)
    n_diff = len(problem._diff_vars)
    dx = np.zeros(n_diff, dtype=float)

    v = np.zeros(n_state_alg + n_diff, dtype=float)
    v[:n_state_alg] = x

    f_sparse = problem._compute_jacobian_sparse_from_S(s_mat.tocsc(), v)
    j_ml = (phi_mat @ f_sparse.T).toarray()
    h = 1e9
    if len(problem._state_eqs) > 0:
        fx = problem.get_j11(x, dx, h).toarray()
        fy = problem.get_j12(x, dx, h).toarray()
        gx = problem.get_j21(x, dx, h).toarray()
        gy = problem.get_j22(x, dx, h).toarray()
        j_sym = np.block([[fx, fy], [gx, gy]])
    else:
        j_sym = problem.get_j22(x, dx, h).toarray()

    # For phasor no-state path, compare against algebraic Jacobian columns only.
    j_ml_view = j_ml[:, :j_sym.shape[1]]
    n_r = min(j_ml_view.shape[0], j_sym.shape[0])
    n_c = min(j_ml_view.shape[1], j_sym.shape[1])
    diff = j_ml_view[:n_r, :n_c] - j_sym[:n_r, :n_c]
    print("\n[Jacobian Reconstruction]")
    print(f"  J_ml shape={j_ml.shape}, J_sym shape={j_sym.shape}, compare_view={j_ml_view.shape}, compare=({n_r},{n_c})")
    print(f"  max|J_ml-J_sym|={np.max(np.abs(diff)):.6e}")
    print(f"  mean|J_ml-J_sym|={np.mean(np.abs(diff)):.6e}")

    # Column-wise diagnostics to identify where mismatch is concentrated.
    all_vars = list(problem._state_vars) + list(problem._algebraic_vars) + list(problem._diff_vars)
    col_max = np.max(np.abs(diff), axis=0)
    row_max = np.max(np.abs(diff), axis=1)

    top_cols = np.argsort(col_max)[-12:][::-1]
    top_rows = np.argsort(row_max)[-12:][::-1]

    print("\n  Top mismatch columns:")
    for c in top_cols:
        if c < len(all_vars):
            vname = str(all_vars[c].name)
        else:
            vname = f"col_{c}"
        abs_col = np.abs(diff[:, c])
        row_peak = int(np.argmax(abs_col))
        print(
            f"    col={int(c):<4} var={vname:<30} max_err={col_max[c]:.6e} at row={row_peak}"
        )

    all_eqs = list(problem._state_eqs) + list(problem._algebraic_eqs)

    print("\n  Top mismatch rows:")
    for r in top_rows:
        abs_row = np.abs(diff[r, :])
        col_peak = int(np.argmax(abs_row))
        if col_peak < len(all_vars):
            col_name = str(all_vars[col_peak].name)
        else:
            col_name = f"col_{col_peak}"
        if r < len(all_eqs):
            eq_txt = str(all_eqs[r]).replace("\n", " ")
        else:
            eq_txt = f"eq_{r}"
        if len(eq_txt) > 140:
            eq_txt = eq_txt[:137] + "..."
        print(
            f"    row={int(r):<4} max_err={row_max[r]:.6e} at col={col_peak} ({col_name})"
        )
        print(f"      eq: {eq_txt}")

    # Inspect exact pair values on worst entries.
    print("\n  Worst entry details (symbolic vs multilinear):")
    flat_order = np.argsort(np.abs(diff), axis=None)[-12:][::-1]

    var_name_to_idx: dict[str, int] = {}
    for i, vv in enumerate(all_vars):
        var_name_to_idx[str(vv.name)] = i

    for k in flat_order:
        rr, cc = np.unravel_index(k, diff.shape)
        if cc < len(all_vars):
            cname = str(all_vars[cc].name)
        else:
            cname = f"col_{cc}"
        jml = float(j_ml_view[rr, cc])
        jsym = float(j_sym[rr, cc])
        print(
            f"    (r={rr}, c={cc:<3} {cname:<28})  J_ml={jml:+.6e}  J_sym={jsym:+.6e}  diff={jml-jsym:+.6e}"
        )

        # Check if multilinear sensitivity moved to a derivative/lifted companion variable.
        companion_names = [f"d_{cname}", f"dt_{cname}", f"dt_1_{cname}"]
        hits: list[str] = []
        for nm in companion_names:
            idx = var_name_to_idx.get(nm)
            if idx is None:
                continue
            val = float(j_ml[rr, idx])
            if abs(val) > 1e-12:
                hits.append(f"{nm}:{val:+.6e}")
        if len(hits) > 0:
            print(f"      companion cols in J_ml -> {'; '.join(hits)}")


def print_E_matrix_comparison(problem, builder: PolynomialMatrixBuilder) -> None:
    x = problem.get_x0()
    n_state = len(problem._state_vars)
    n_alg = len(problem._algebraic_vars)

    v_dict: dict[str, float] = {}
    vars_sa = list(problem._state_vars) + list(problem._algebraic_vars)
    for i, vv in enumerate(vars_sa):
        v_dict[str(vv.name)] = float(x[i])
    for dv in problem._diff_vars:
        v_dict[str(dv.name)] = 0.0

    builder.linearize(v_dict)
    if builder.E is None:
        print("\n[E Matrix Comparison]")
        print("  builder.E is None")
        return

    e_ml, _, _ = problem.linearize_multilinear(x=x)
    e_pb = builder.E.toarray()

    n_r = min(e_ml.shape[0], e_pb.shape[0])
    n_c = min(e_ml.shape[1], e_pb.shape[1])
    diff = e_ml[:n_r, :n_c] - e_pb[:n_r, :n_c]

    print("\n[E Matrix Comparison]")
    print(f"  E_ml shape={e_ml.shape}, E_pb shape={e_pb.shape}, compare=({n_r},{n_c})")
    print(f"  max|E_ml-E_pb|={np.max(np.abs(diff)):.6e}")
    print(f"  mean|E_ml-E_pb|={np.mean(np.abs(diff)):.6e}")

    absd = np.abs(diff)
    total = absd.size
    close = int(np.sum(absd <= 1e-9))
    print(f"  |diff|<=1e-9: {close}/{total}")

    vars_sa = list(problem._state_vars) + list(problem._algebraic_vars)
    row_max = np.max(absd, axis=1)
    col_max = np.max(absd, axis=0)
    top_rows = np.argsort(row_max)[-12:][::-1]
    top_cols = np.argsort(col_max)[-12:][::-1]

    print("\n  Top E mismatch rows:")
    all_eqs = list(problem._state_eqs) + list(problem._algebraic_eqs)
    for r in top_rows:
        c_peak = int(np.argmax(absd[r, :]))
        c_name = str(vars_sa[c_peak].name) if c_peak < len(vars_sa) else f"col_{c_peak}"
        print(f"    eq={int(r):<4} max_err={row_max[r]:.6e} at col={c_peak} ({c_name})")
        if r < len(all_eqs):
            eq_txt = str(all_eqs[r]).replace("\n", " ")
            if len(eq_txt) > 180:
                eq_txt = eq_txt[:177] + "..."
            print(f"      eq: {eq_txt}")

    print("\n  Top E mismatch cols:")
    for c in top_cols:
        vname = str(vars_sa[c].name) if c < len(vars_sa) else f"col_{c}"
        r_peak = int(np.argmax(absd[:, c]))
        r_name = str(vars_sa[r_peak].name) if r_peak < len(vars_sa) else f"row_{r_peak}"
        print(f"    col={int(c):<4} var={vname:<30} max_err={col_max[c]:.6e} at row={r_peak} ({r_name})")

    print("\n  Worst E entry details (pb vs ml):")
    flat_order = np.argsort(absd, axis=None)[-12:][::-1]
    for k in flat_order:
        rr, cc = np.unravel_index(k, absd.shape)
        c_name = str(vars_sa[cc].name) if cc < len(vars_sa) else f"col_{cc}"
        vml = float(e_ml[rr, cc])
        vpb = float(e_pb[rr, cc])
        print(
            f"    (eq={rr:<3}, c={cc:<3} {c_name:<28}) E_ml={vml:+.6e} E_pb={vpb:+.6e} diff={vml-vpb:+.6e}"
        )


def print_get_E_projection_comparison(problem) -> None:
    x = problem.get_x0()
    dx = np.zeros(len(problem._diff_vars), dtype=float)

    all_eqs = list(problem._state_eqs) + list(problem._algebraic_eqs)
    xdot = list(problem._diff_vars)
    e_call = SymbolicJacobian(
        eqs=all_eqs,
        variables=xdot,
        compiler_names_dict=problem._compiler_names_dict,
        alias_names_dict=problem._alias_names_dict,
        VARS_NAME=problem.VARS_NAME,
        DIFF_NAME=problem.DIFF_NAME,
        EVENT_PARAMS_NAME=problem.VARIABLE_PARAMS_NAME,
        PARAMS_NAME=problem.CONSTANT_PARAMS_NAME,
        static=True,
    )
    e_partial = e_call(x, dx, problem._variable_parameters_values, problem._constant_params, h=0).toarray()

    n_vars = problem._n_vars
    e_proj = np.zeros((n_vars, n_vars), dtype=float)
    for j, dvar in enumerate(xdot):
        base_var = dvar.base_var
        col_idx = problem._uid2idx_vars.get(base_var.uid, None)
        if col_idx is None:
            continue
        e_proj[:, col_idx] += e_partial[:, j]
    n_states = problem.get_states_number()
    e_proj[:n_states, :n_states] -= np.eye(n_states, dtype=e_proj.dtype)

    e_get = problem.get_E_matrix(x, dx)
    e_ml, _, _ = problem.linearize_multilinear(x=x)

    n_r = min(e_get.shape[0], e_ml.shape[0], e_proj.shape[0])
    n_c = min(e_get.shape[1], e_ml.shape[1], e_proj.shape[1])

    diff_get_proj = e_get[:n_r, :n_c] - e_proj[:n_r, :n_c]
    diff_ml_get = e_ml[:n_r, :n_c] - e_get[:n_r, :n_c]

    print("\n[E Projection Cross-check]")
    print(f"  compare shape=({n_r},{n_c})")
    print(f"  max|E_get - E_proj|={np.max(np.abs(diff_get_proj)):.6e}")
    print(f"  mean|E_get - E_proj|={np.mean(np.abs(diff_get_proj)):.6e}")
    print(f"  max|E_ml - E_get|={np.max(np.abs(diff_ml_get)):.6e}")
    print(f"  mean|E_ml - E_get|={np.mean(np.abs(diff_ml_get)):.6e}")


def print_fd_diffvar_check(problem, eps: float = 1e-7) -> None:
    x = problem.get_x0()
    n_diff = len(problem._diff_vars)
    dx0 = np.zeros(n_diff, dtype=float)

    all_eqs = list(problem._state_eqs) + list(problem._algebraic_eqs)
    xdot = list(problem._diff_vars)
    e_call = SymbolicJacobian(
        eqs=all_eqs,
        variables=xdot,
        compiler_names_dict=problem._compiler_names_dict,
        alias_names_dict=problem._alias_names_dict,
        VARS_NAME=problem.VARS_NAME,
        DIFF_NAME=problem.DIFF_NAME,
        EVENT_PARAMS_NAME=problem.VARIABLE_PARAMS_NAME,
        PARAMS_NAME=problem.CONSTANT_PARAMS_NAME,
        static=True,
    )
    e_partial = e_call(x, dx0, problem._variable_parameters_values, problem._constant_params, h=0).toarray()

    def residual(dx_vec: np.ndarray) -> np.ndarray:
        if len(problem._state_eqs) > 0:
            f = np.asarray(problem.rhs_state(x, dx_vec), dtype=float)
        else:
            f = np.zeros(0, dtype=float)
        g = np.asarray(problem.rhs_algebraic(x, dx_vec), dtype=float)
        return np.concatenate([f, g])

    n_eq = e_partial.shape[0]
    j_fd = np.zeros((n_eq, n_diff), dtype=float)
    for j in range(n_diff):
        dxp = dx0.copy()
        dxm = dx0.copy()
        dxp[j] += eps
        dxm[j] -= eps
        rp = residual(dxp)
        rm = residual(dxm)
        j_fd[:, j] = (rp - rm) / (2.0 * eps)

    diff = j_fd - e_partial
    absd = np.abs(diff)
    print("\n[Finite-Difference dR/ddx Check]")
    print(f"  fd shape={j_fd.shape}, symbolic shape={e_partial.shape}")
    print(f"  max|fd-symbolic|={np.max(absd):.6e}")
    print(f"  mean|fd-symbolic|={np.mean(absd):.6e}")

    col_max = np.max(absd, axis=0)
    top_cols = np.argsort(col_max)[-8:][::-1]
    print("  Top diff-var column mismatches:")
    for j in top_cols:
        name = str(xdot[j].name)
        print(f"    j={int(j):<3} dvar={name:<30} max_err={col_max[j]:.6e}")


def main() -> None:
    problem = create_problem(hard_sat_type="ml")

    s_ml, phi_ml = problem.build_multilinear_matrices()
    print("[RmsProblemMultilinear]")
    print("  S:", matrix_fingerprint(s_ml))
    print("  Phi:", matrix_fingerprint(phi_ml))
    print_monomial_samples("RmsProblemMultilinear", s_ml, phi_ml)
    print_rejected_terms(problem)
    print_residual_reconstruction(problem, s_ml, phi_ml)
    print_jacobian_reconstruction(problem, s_ml, phi_ml)

    builder = PolynomialMatrixBuilder(problem=problem, verbose=False)
    s_pb, phi_pb = builder.S_H, builder.Phi_H
    print("[PolynomialMatrixBuilder]")
    print("  S_H:", matrix_fingerprint(s_pb))
    print("  Phi_H:", matrix_fingerprint(phi_pb))
    print_monomial_samples("PolynomialMatrixBuilder", s_pb, phi_pb)

    same_s_shape = s_ml.shape == s_pb.shape
    same_phi_shape = phi_ml.shape == phi_pb.shape
    print("[Comparison]")
    print(f"  S same shape: {same_s_shape}")
    print(f"  Phi same shape: {same_phi_shape}")
    print_E_matrix_comparison(problem, builder)
    print_get_E_projection_comparison(problem)
    print_fd_diffvar_check(problem)

    out_dir = project_base / "trunk" / "tensygrid" / "debug_matrices"
    os.makedirs(out_dir, exist_ok=True)
    #sparse.save_npz(str(out_dir / "S_multilinear.npz"), s_ml)
    #sparse.save_npz(str(out_dir / "Phi_multilinear.npz"), phi_ml)
    #sparse.save_npz(str(out_dir / "S_polybuilder.npz"), s_pb)
    #sparse.save_npz(str(out_dir / "Phi_polybuilder.npz"), phi_pb)
    #print(f"  Saved matrices to: {out_dir}")


if __name__ == "__main__":
    main()
