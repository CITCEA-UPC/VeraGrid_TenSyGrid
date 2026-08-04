# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Utilities for CPN1 (Companion Power Network) steady-state computation.

The CPN1 system solves for the steady-state operating point of a power system
including generator controllers (exciters, governors, limiters). Unlike power
flow which only solves P(V,θ)=0, Q(V,θ)=0 for bus voltages, CPN1 solves the
full differential-algebraic system with all time derivatives set to zero.

The system is represented as: Phi @ varphi(v) = 0

where:
    S: (n_vars, n_mon) — maps variables to monomials
    Phi: (n_eqs, n_mon) — equation coefficients
    varphi_j(v) = prod_p (s_{i_p,j} * v_{i_p} + (1 - |s_{i_p,j}|))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import scipy
from scipy import sparse
import numpy as np

from VeraGridEngine.Utils.Symbolic.symbolic import Const

project_base = Path(__file__).resolve().parents[2]


def stable_rank(A, method='svd'):
    """Compute the numerical rank of a matrix using a stable method.

    Two methods are available:
        - 'svd': SVD-based, equivalent to MATLAB's ``rank(A)``.
          Tolerance = max(m,n) * eps * sigma_max.
        - 'qr': Rank-Revealing QR (RRQR) with column pivoting.
          Faster for large sparse matrices. Uses the same tolerance formula
          applied to the diagonal of R.

    Args:
        A: Input matrix (dense or sparse).
        method: 'svd' (default) or 'qr'.

    Returns:
        Numerical rank (int).
    """
    import scipy.sparse as sp

    if sp.issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = np.asarray(A, dtype=float)

    m, n = A_dense.shape
    tol_default = max(m, n) * np.finfo(float).eps

    if method == 'svd':
        s = np.linalg.svd(A_dense, compute_uv=False)
        tol = tol_default * s[0] if s.size > 0 and s[0] > 0 else tol_default
        return int(np.sum(s > tol))

    elif method == 'qr':
        from scipy.linalg import qr
        _, R, _ = qr(A_dense, pivoting=True)
        r_diag = np.abs(np.diag(R))
        tol = tol_default * r_diag[0] if r_diag.size > 0 and r_diag[0] > 0 else tol_default
        return int(np.sum(r_diag > tol))

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'svd' or 'qr'.")

import VeraGridEngine.api as vge
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model


def ensure_unique_device_names(grid) -> None:
    """Rename devices to ensure unique names within each category."""
    def _rename(devices, prefix: str) -> None:
        seen: dict[str, int] = {}
        for i, dev in enumerate(devices):
            base = str(dev.name).strip() if getattr(dev, "name", None) else f"{prefix}_{i}"
            if base not in seen:
                seen[base] = 0
                dev.name = base
            else:
                seen[base] += 1
                dev.name = f"{base}_{seen[base]}"

    _rename(list(grid.generators), "gen")
    _rename(list(grid.lines), "line")
    _rename(list(grid.transformers2w), "trafo")
    _rename(list(grid.loads), "load")
    _rename(list(grid.shunts), "shunt")


def _ideal_vsource_phasor_template(vfactory, name="IdealVSource"):
    """Ideal AC voltage source in phasor coordinates (Vr, Vi), no dynamics.

    Uses the bus Vr/Vi directly (no own algebraic vars) to avoid duplicate
    registration in the multilinear problem.
    """
    from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
    from VeraGridEngine.Utils.Symbolic.block import Block
    templ = RmsModelTemplate()
    templ.tpe = vge.DeviceType.ExternalGridDevice
    Vr = vfactory.add_var("Vr_src", vge.VarPowerFlowReferenceType.Vr)
    Vi = vfactory.add_var("Vi_src", vge.VarPowerFlowReferenceType.Vi)
    templ.block = Block(
        algebraic_eqs=[Vr - 1.0, Vi - 0.0],
        algebraic_vars=[],
    )
    templ.block.in_vars = [Vr, Vi]
    templ.block.name = name
    templ.block.external_mapping = {
        vge.VarPowerFlowReferenceType.Vr: Vr,
        vge.VarPowerFlowReferenceType.Vi: Vi,
    }
    return templ


def _ideal_vsource_dc_template(vfactory, name="IdealDCSource"):
    """Ideal DC voltage source, no dynamics. Imposes Vdc = 1.0."""
    from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
    from VeraGridEngine.Utils.Symbolic.block import Block
    templ = RmsModelTemplate()
    templ.tpe = vge.DeviceType.ExternalGridDevice
    Vdc = vfactory.add_var("Vdc_src", vge.VarPowerFlowReferenceType.Vdc)
    templ.block = Block(
        algebraic_eqs=[Vdc - 1.0],
        algebraic_vars=[],
    )
    templ.block.in_vars = [Vdc]
    templ.block.name = name
    templ.block.external_mapping = {
        vge.VarPowerFlowReferenceType.Vdc: Vdc,
    }
    return templ


def build_problems(grid_filename: str, grid=None):
    """Load grid, initialize RMS models, run power flow, and build the multilinear problem."""
    if grid is None:
        grid_path = project_base / "Grids_and_profiles" / "grids" / grid_filename
        grid = vge.open_file(str(grid_path))
    ensure_unique_device_names(grid)

    for bus in grid.buses:
        if bus.rms_model.empty():
            vge.initialize_bus_phasor_rms(bus, vf=grid.var_factory)

    for igen, gen in enumerate(grid.generators):
        if not gen.active:
            continue
        if not gen.rms_model.empty():
            continue
        if gen.bus.is_dc:
            from VeraGridEngine.Templates.Rms.dc_voltage_source import DCVoltageSource
            gen_mdl = DCVoltageSource(grid.var_factory, Vdc=gen.bus.rms_model.out_vars[0], name=gen.name).block
            gen_mdl = vge.to_implicit(gen_mdl, grid.var_factory)
            set_rms_model(device=gen, model=gen_mdl, var_factory=grid.var_factory)
            continue
        gen_mdl = vge.get_complete_generator_template_phasor(grid.var_factory, name=f"Gen{igen}").block
        grid.var_factory.add_connections([gen_mdl.in_vars[0]], [gen.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([gen_mdl.in_vars[1]], [gen.bus.rms_model.out_vars[1]])
        gen_mdl = vge.to_implicit(gen_mdl, grid.var_factory)
        set_rms_model(device=gen, model=gen_mdl, var_factory=grid.var_factory)

    for line in grid.lines:
        if not line.active:
            continue
        if not line.rms_model.empty():
            continue
        line_mdl = vge.get_line_phasor_rms_template(grid.var_factory, name=line.name).block
        grid.var_factory.add_connections([line_mdl.in_vars[0]], [line.bus_from.rms_model.out_vars[0]])
        grid.var_factory.add_connections([line_mdl.in_vars[1]], [line.bus_from.rms_model.out_vars[1]])
        grid.var_factory.add_connections([line_mdl.in_vars[2]], [line.bus_to.rms_model.out_vars[0]])
        grid.var_factory.add_connections([line_mdl.in_vars[3]], [line.bus_to.rms_model.out_vars[1]])
        line_mdl = vge.to_implicit(line_mdl, grid.var_factory)
        set_rms_model(device=line, model=line_mdl, var_factory=grid.var_factory)

    for load in grid.loads:
        if not load.active:
            continue
        if not load.rms_model.empty():
            continue
        load_mdl = vge.get_load_phasor_current_rms_template(grid.var_factory, name=load.name).block
        grid.var_factory.add_connections([load_mdl.in_vars[0]], [load.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([load_mdl.in_vars[1]], [load.bus.rms_model.out_vars[1]])
        set_rms_model(device=load, model=load_mdl, var_factory=grid.var_factory)

    for trafo in grid.transformers2w:
        if not trafo.active:
            continue
        if not trafo.rms_model.empty():
            continue
        trafo_mdl = vge.initialize_trafo_rms(trafo, grid.var_factory, use_phasor_template=True).block
        grid.var_factory.add_connections([trafo_mdl.in_vars[0]], [trafo.bus_from.rms_model.out_vars[0]])
        grid.var_factory.add_connections([trafo_mdl.in_vars[1]], [trafo.bus_from.rms_model.out_vars[1]])
        grid.var_factory.add_connections([trafo_mdl.in_vars[2]], [trafo.bus_to.rms_model.out_vars[0]])
        grid.var_factory.add_connections([trafo_mdl.in_vars[3]], [trafo.bus_to.rms_model.out_vars[1]])
        trafo_mdl = vge.to_implicit(trafo_mdl, grid.var_factory)
        set_rms_model(device=trafo, model=trafo_mdl, var_factory=grid.var_factory)

    for shunt in grid.shunts:
        if not shunt.active:
            continue
        if not shunt.rms_model.empty():
            continue
        shunt_mdl = vge.get_shunt_template(grid.var_factory, name=shunt.name, phasor=True).block
        grid.var_factory.add_connections([shunt_mdl.in_vars[0]], [shunt.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([shunt_mdl.in_vars[1]], [shunt.bus.rms_model.out_vars[1]])
        shunt_mdl = vge.to_implicit(shunt_mdl, grid.var_factory)
        set_rms_model(device=shunt, model=shunt_mdl, var_factory=grid.var_factory)

    for eg in grid.external_grids:
        if not eg.active:
            continue
        if not eg.rms_model.empty():
            continue
        if eg.bus.is_dc:
            eg_mdl = _ideal_vsource_dc_template(grid.var_factory, name=eg.name).block
            grid.var_factory.add_connections([eg_mdl.in_vars[0]], [eg.bus.rms_model.out_vars[0]])
        else:
            eg_mdl = _ideal_vsource_phasor_template(grid.var_factory, name=eg.name).block
            grid.var_factory.add_connections([eg_mdl.in_vars[0]], [eg.bus.rms_model.out_vars[0]])
            grid.var_factory.add_connections([eg_mdl.in_vars[1]], [eg.bus.rms_model.out_vars[1]])
        eg_mdl = vge.to_implicit(eg_mdl, grid.var_factory)
        set_rms_model(device=eg, model=eg_mdl, var_factory=grid.var_factory)

    pf_results = vge.power_flow(grid, vge.PowerFlowOptions(solver_type=vge.SolverType.NR, tolerance=1e-5, max_iter=30))
    if not pf_results.converged:
        raise RuntimeError("Power flow did not converge")

    # Fix DC bus voltages: PF gives Vm=0 for DC buses, but we need Vdc=1.0
    for i, bus in enumerate(grid.buses):
        if bus.is_dc:
            pf_results.voltage[i] = 1.0 + 0.0j

    rms_options_ml = vge.RmsOptions(
        time_step=0.01,
        simulation_time=1.0,
        tolerance=1e-6,
        max_iter=20,
        problem_type=vge.RmsProblemTypes.Multilinear,
    )

    problem_ml = vge.RmsProblemMultilinear(grid=grid, options=rms_options_ml, pf_results=pf_results)
    return problem_ml, pf_results, rms_options_ml


@dataclass
class CpnSystem:
    """Result of CPN1 system processing.

    Attributes:
        S: Variable-to-monomial mapping (n_vars, n_mon)
        Phi: Equation coefficients (n_eqs, n_mon)
        vars_list: Variable names (possibly with uid suffixes for deduplication)
        eqs_list: Equation strings (for inspection/export)
        orig_keep_idx: Maps local variable indices back to original problem variables
        x0: Initial guess from power flow (full length)
    """
    S: sparse.csc_matrix
    Phi: sparse.csr_matrix
    vars_list: list[str]
    eqs_list: list[str]
    orig_keep_idx: np.ndarray
    x0: np.ndarray


def build_var_names(problem_ml) -> tuple[list[str], dict[int, str]]:
    """Build unique variable names with uid-based suffixes for duplicates.

    Returns:
        vars_list: Ordered variable names
        uid_to_name: Mapping from variable uid to unique name
    """
    variables = problem_ml.state_and_algebraic_vars
    vars_list: list[str] = []
    uid_to_name: dict[int, str] = {}
    name_seen: dict[str, int] = {}
    for var in variables:
        name = str(var)
        n = name_seen.get(name, 0)
        name_seen[name] = n + 1
        if n == 0:
            vars_list.append(name)
            uid_to_name[var.uid] = name
        else:
            uid_suffix = var.uid if hasattr(var, 'uid') else n
            suffixed = f"{name}_{uid_suffix}"
            vars_list.append(suffixed)
            uid_to_name[var.uid] = suffixed
    return vars_list, uid_to_name


def build_eqs_list(all_raw_eqs, uid_to_name, deduplicate=True):
    """Convert symbolic equations to strings with unique variable names.

    Args:
        all_raw_eqs: List of symbolic equations
        uid_to_name: Variable name mapping from build_var_names
        deduplicate: If True, remove equations with identical (expression, variables) signature

    Returns:
        eqs_list: List of equation strings
        keep_eq_mask: Boolean mask indicating which original equations were kept
    """
    eqs_list: list[str] = []
    keep_eq_mask = np.ones(len(all_raw_eqs), dtype=bool)

    if deduplicate:
        eq_sig_seen: set[tuple[str, frozenset[int]]] = set()
        for i, eq in enumerate(all_raw_eqs):
            eq_vars = eq.get_vars()
            var_uids = frozenset(v.uid for v in eq_vars)
            sig = (str(eq), var_uids)
            if sig in eq_sig_seen:
                keep_eq_mask[i] = False
                continue
            eq_sig_seen.add(sig)
            eq_str = _rename_vars_in_expr(eq, uid_to_name)
            eqs_list.append(eq_str)
    else:
        for eq in all_raw_eqs:
            eq_str = _rename_vars_in_expr(eq, uid_to_name)
            eqs_list.append(eq_str)

    return eqs_list, keep_eq_mask


def _rename_vars_in_expr(eq, uid_to_name: dict[int, str]) -> str:
    """Rename variables in an expression tree to unique names and return the string.

    Uses the expression tree's :meth:`subs` to avoid string-level ambiguity when
    multiple variables share the same ``str(v)`` (e.g. two buses both named ``Vr``).
    """
    from VeraGridEngine.Utils.Symbolic.symbolic import Var
    mapping = {}
    for v in eq.get_vars():
        new_name = uid_to_name.get(v.uid)
        if new_name is not None and new_name != v.name:
            mapping[v] = Var(name=new_name, reference=v.ref, uid=v.uid)
    if mapping:
        eq = eq.subs(mapping)
    return str(eq)


def regenerate_eqs_list(S, Phi, vars_list):
    """Rebuild equation strings from S, Phi, and vars_list.

    After the pipeline drops or substitutes variables, the original eqs_list
    strings can reference variables that no longer exist in S. This function
    reconstructs each equation as a sum of monomial terms directly from the
    sparse matrices, guaranteeing consistency with S and Phi.
    """
    S_csc = S.tocsc() if S.shape[1] > 0 else S.tocsc()
    n_eqs = Phi.shape[0]
    n_mon = S.shape[1]
    eqs = []

    for i in range(n_eqs):
        terms = []
        phi_row = Phi.getrow(i)
        mon_cols = phi_row.nonzero()[1]

        for j in mon_cols:
            phi_coeff = float(Phi[i, j])
            col_j = S_csc.getcol(j)
            var_rows = col_j.nonzero()[0]

            if len(var_rows) == 0:
                terms.append(f"({phi_coeff})")
                continue

            factors = []
            for vr in var_rows:
                vname = vars_list[int(vr)]
                exp = col_j[vr, 0]
                if exp == 1.0:
                    factors.append(f"({vname})")
                else:
                    factors.append(f"({vname}**{exp})")

            mono_str = " * ".join(factors)
            if len(factors) == 1:
                terms.append(f"({phi_coeff}) * {mono_str}")
            else:
                terms.append(f"({phi_coeff}) * {mono_str}")

        eqs.append(" + ".join(terms) if terms else "0")

    return eqs


def cleanup_system(S, Phi, vars_list, eqs_list, orig_keep_idx):
    """Remove orphaned monomial columns and variable rows iteratively.

    - Monomial columns with all-zero Phi rows (not in any equation)
    - Monomial columns with all-zero S rows (not used by any variable)
    - Variable rows with all-zero S rows (not in any monomial)

    Iterates because removing columns can orphan rows and vice-versa.
    """
    while True:
        changed = False
        S.eliminate_zeros()
        Phi.eliminate_zeros()

        empty_phi_cols = np.asarray(Phi.getnnz(axis=0)).flatten() == 0
        if np.any(empty_phi_cols):
            keep = np.where(~empty_phi_cols)[0]
            S = S[:, keep]
            Phi = Phi[:, keep]
            changed = True

        empty_s_cols_no_phi = (np.asarray(S.getnnz(axis=0)).flatten() == 0) & (np.asarray(Phi.getnnz(axis=0)).flatten() == 0)
        if np.any(empty_s_cols_no_phi):
            keep = np.where(~empty_s_cols_no_phi)[0]
            S = S[:, keep]
            Phi = Phi[:, keep]
            changed = True

        empty_s_rows = np.asarray(S.getnnz(axis=1)).flatten() == 0
        if np.any(empty_s_rows):
            keep = np.where(~empty_s_rows)[0]
            S = S[keep, :]
            vars_list = [vars_list[i] for i in keep]
            orig_keep_idx = orig_keep_idx[keep]
            changed = True

        if not changed:
            break

    return S, Phi, vars_list, eqs_list, orig_keep_idx


def simplify_system(S, Phi, vars_list, eqs_list, orig_keep_idx):
    """Remove trivially satisfied or redundant equations iteratively.

    Detects and removes:
    - Bare variable names (= 0)
    - Negated variable names (= 0)
    - (Var) - (0...) patterns (= 0)
    - Bare numeric constants (trivially satisfied)
    - Empty Phi rows (no monomials)

    After removing equation rows, also removes orphaned monomial columns
    and variable rows that no longer participate in any monomial.

    Iterates up to 50 times since removing one equation can reveal others
    as trivial.

    Returns:
        Simplified S, Phi, vars_list, eqs_list, orig_keep_idx
    """
    import re

    def _is_identifier(s):
        return s.isidentifier() if s else False

    def _extract_var_name(eq_str):
        s = eq_str.strip()
        m = re.match(r'^-?([A-Za-z_]\w*)$', s)
        if m:
            return m.group(1)
        m = re.match(r'^-?\(([A-Za-z_]\w*)\)$', s)
        if m:
            return m.group(1)
        if s.startswith('(') and ') - (0' in s and s.endswith(')'):
            return s.split(') - ')[0][1:]
        if s.startswith('-(') and ') - (0' in s and s.endswith(')'):
            return s.split(') - ')[0][2:]
        return None

    for _ in range(50):
        eq_drop: set[int] = set()
        var_drop: set[int] = set()

        for i, eq_str in enumerate(eqs_list):
            s = eq_str.strip()
            is_zero_var = False
            is_trivial = False

            if re.match(r'^[A-Za-z_]\w*$', s):
                is_zero_var = True
            elif re.match(r'^-[A-Za-z_]\w*$', s):
                is_zero_var = True
            elif re.match(r'^\(([A-Za-z_]\w*)\)$', s):
                is_zero_var = True
            elif re.match(r'^-\(([A-Za-z_]\w*)\)$', s):
                is_zero_var = True
            elif s.startswith('(') and ') - (0' in s and s.endswith(')'):
                vn = s.split(') - ')[0][1:]
                rp = s.split(' - (')[1][:-1]
                if (rp == '0' or rp.startswith('0.')) and _is_identifier(vn):
                    is_zero_var = True
            elif s.startswith('-(') and ') - (0' in s and s.endswith(')'):
                vn = s.split(') - ')[0][2:]
                rp = s.split(' - (')[1][:-1]
                if (rp == '0' or rp.startswith('0.')) and _is_identifier(vn):
                    is_zero_var = True
            elif re.match(r'^-?\d+\.?\d*(e[+-]?\d+)?$', s):
                is_trivial = True

            if is_zero_var:
                eq_drop.add(i)
                vn = _extract_var_name(eq_str)
                if vn and vn in vars_list:
                    var_drop.add(vars_list.index(vn))
            elif is_trivial:
                eq_drop.add(i)

        zero_phi_rows = set(np.where(np.asarray(Phi.getnnz(axis=1)).flatten() == 0)[0])
        eq_drop |= zero_phi_rows

        if not eq_drop and not var_drop:
            break

        if eq_drop:
            eq_keep = np.array(sorted(set(range(Phi.shape[0])) - eq_drop))
            Phi = Phi[eq_keep, :]
            eqs_list = [eqs_list[i] for i in eq_keep]

        if var_drop:
            S_csc = S.tocsc()
            for vi in var_drop:
                monom_cols = S_csc.getrow(vi).nonzero()[1]
                for j in monom_cols:
                    s_ij = S_csc[vi, j]
                    alpha = 1.0 - abs(s_ij)
                    if alpha != 1.0:
                        Phi[:, j] = Phi[:, j] * alpha
            var_keep = np.array(sorted(set(range(S.shape[0])) - var_drop))
            S = S[var_keep, :]
            vars_list = [vars_list[i] for i in var_keep]
            orig_keep_idx = orig_keep_idx[var_keep]

    return cleanup_system(S, Phi, vars_list, eqs_list, orig_keep_idx)


def substitute_pinned_constants(S, Phi, vars_list, eqs_list, orig_keep_idx):
    """Substitute variables pinned to constant values and remove their equations.

    Detects equations of the form ``var - constant`` (with or without parentheses,
    including negated forms like ``-(var) - (constant)``), substitutes the constant
    value into every monomial that contains the variable, and removes both the
    equation row (from Phi) and the variable row (from S).

    Returns:
        Updated S, Phi, vars_list, eqs_list, orig_keep_idx
    """
    import re

    _NUM = r'(-?\d+\.?\d*(?:e[+-]?\d+)?)'
    _VAR = r'([A-Za-z_]\w*)'

    patterns = [
        (re.compile(r'^' + _VAR + r'\s+-\s+' + _NUM + r'$'), 1),
        (re.compile(r'^\(' + _VAR + r'\)\s+-\s+\(' + _NUM + r'\)$'), 1),
        (re.compile(r'^\(' + _VAR + r'\)\s+-\s+' + _NUM + r'$'), 1),
        (re.compile(r'^-\(' + _VAR + r'\)\s+-\s+\(' + _NUM + r'\)$'), -1),
        (re.compile(r'^-\(' + _VAR + r'\)\s+-\s+' + _NUM + r'$'), -1),
    ]

    pinned: list[tuple[int | None, float, int]] = []
    eq_only_drop: set[int] = set()

    for i, eq_str in enumerate(eqs_list):
        s = eq_str.strip()
        for pat, sign in patterns:
            m = pat.match(s)
            if m:
                var_name = m.group(1)
                const_val = sign * float(m.group(2))
                if var_name in vars_list:
                    var_idx = vars_list.index(var_name)
                    pinned.append((var_idx, const_val, i))
                else:
                    eq_only_drop.add(i)
                break

    if not pinned and not eq_only_drop:
        return S, Phi, vars_list, eqs_list, orig_keep_idx

    S_csc = S.tocsc()
    eq_rows_to_drop: set[int] = set()

    for var_idx, const_val, eq_idx in pinned:
        monom_cols = S_csc.getrow(var_idx).nonzero()[1]
        for j in monom_cols:
            s_ij = S_csc[var_idx, j]
            alpha = s_ij * const_val + (1.0 - abs(s_ij))
            if alpha != 1.0:
                Phi[:, j] = Phi[:, j] * alpha
        eq_rows_to_drop.add(eq_idx)

    eq_rows_to_drop |= eq_only_drop

    eq_keep = np.array([i for i in range(Phi.shape[0]) if i not in eq_rows_to_drop])
    Phi = Phi[eq_keep, :]
    eqs_list = [eqs_list[i] for i in eq_keep]

    var_drop = {pv[0] for pv in pinned}
    var_keep = np.array([i for i in range(S.shape[0]) if i not in var_drop])
    S = S[var_keep, :]
    vars_list = [vars_list[i] for i in var_keep]
    orig_keep_idx = orig_keep_idx[var_keep]

    return cleanup_system(S, Phi, vars_list, eqs_list, orig_keep_idx)


def handle_free_variables(S, Phi, vars_list, eqs_list, orig_keep_idx, x0):
    """Handle variables that don't appear in any monomial (free variables).

    A variable is "free" if its row in S is all zeros — it doesn't participate
    in any monomial. These variables need to be pinned or removed:

    - Nonzero free variables: pinned to their x0 value by adding equations
      (var - x0[i] = 0) to the system
    - Zero free variables: removed from the system entirely

    Returns:
        Updated S, Phi, vars_list, eqs_list, orig_keep_idx
    """
    free_mask = S.getnnz(axis=1) == 0
    free_local = np.where(free_mask)[0]
    zero_var_rows: set[int] = set()

    if len(free_local) > 0:
        nonzero_free = [li for li in free_local if not np.isclose(x0[orig_keep_idx[li]], 0)]

        if nonzero_free:
            S_csc = S.tocsc()
            const_col = next((j for j in range(S.shape[1]) if S_csc.indptr[j] == S_csc.indptr[j + 1]), None)
            if const_col is None:
                const_col = S.shape[1]
                S = sparse.hstack([S, sparse.csc_matrix((S.shape[0], 1))], format='csc')
                Phi = sparse.hstack([Phi, sparse.csr_matrix((Phi.shape[0], 1))], format='csr')

            for li in nonzero_free:
                oi = orig_keep_idx[li]
                var_name = vars_list[li]
                eq_col = S.shape[1]
                col_data = sparse.csc_matrix(([1.0], ([li], [0])), shape=(S.shape[0], 1))
                S = sparse.hstack([S, col_data], format='csc')
                Phi = sparse.hstack([Phi, sparse.csr_matrix((Phi.shape[0], 1))], format='csr')
                eq_row = sparse.csr_matrix(([1.0, -x0[oi]], ([0, 0], [eq_col, const_col])), shape=(1, S.shape[1]))
                Phi = sparse.vstack([Phi, eq_row], format='csr')
                eqs_list.append(f"{var_name} - {x0[oi]}")

        zero_var_rows = set(free_local) - set(nonzero_free)

    if zero_var_rows:
        keep = np.array([i for i in range(S.shape[0]) if i not in zero_var_rows])
        S = S[keep, :]
        vars_list = [vars_list[i] for i in keep]
        orig_keep_idx = orig_keep_idx[keep]

    return S, Phi, vars_list, eqs_list, orig_keep_idx


def pin_vars_without_trivial_monomials(S, Phi, vars_list, eqs_list, orig_keep_idx, x0):
    """Substitute variables that lost their trivial monomial with their x0 value.

    After simplification, some variables may not have a trivial monomial (a
    monomial column where only that variable has a non-zero S entry). These
    variables are only constrained through shared product monomials, making
    the system underdetermined (n_vars > n_mon).

    This function pins those variables to their x0 value by substituting them
    into every monomial they participate in (scaling Phi columns), then removing
    the variable rows from S. This is the same approach as
    ``substitute_pinned_constants`` but triggered by structural analysis
    (missing trivial monomial) rather than equation pattern matching.
    """
    n_vars, n_mon = S.shape
    if n_vars <= n_mon:
        return S, Phi, vars_list, eqs_list, orig_keep_idx

    S_csc = S.tocsc()

    has_trivial = np.zeros(n_vars, dtype=bool)
    for j in range(n_mon):
        start = S_csc.indptr[j]
        end = S_csc.indptr[j + 1]
        if end - start == 1:
            has_trivial[S_csc.indices[start]] = True

    orphaned = np.where(~has_trivial)[0]
    if len(orphaned) == 0:
        return S, Phi, vars_list, eqs_list, orig_keep_idx

    S_csc = S.tocsc()
    for vi in orphaned:
        const_val = float(x0[orig_keep_idx[vi]])
        monom_cols = S_csc.getrow(vi).nonzero()[1]
        for j in monom_cols:
            s_ij = S_csc[vi, j]
            alpha = s_ij * const_val + (1.0 - abs(s_ij))
            if alpha != 1.0:
                Phi[:, j] = Phi[:, j] * alpha

    var_keep = np.array(sorted(set(range(n_vars)) - set(orphaned)))
    S = S[var_keep, :]
    vars_list = [vars_list[i] for i in var_keep]
    orig_keep_idx = orig_keep_idx[var_keep]

    return cleanup_system(S, Phi, vars_list, eqs_list, orig_keep_idx)


def reduce_rank_qr(S, Phi, vars_list, eqs_list, orig_keep_idx, x0_full):
    """Remove linearly dependent equations using QR column pivoting.

    Uses QR decomposition with column pivoting on J^T to identify the
    maximum set of linearly independent equations. Removes all dependent
    equation rows from Phi regardless of whether n_eqs > n_vars or not,
    since rank deficiency is the real concern.

    Only removes equation rows from Phi — never removes variable rows from S,
    as that would change the monomial structure and alter the system.

    Returns:
        Updated S, Phi, vars_list, eqs_list, orig_keep_idx
    """
    n_eqs = Phi.shape[0]
    if n_eqs == 0:
        return S, Phi, vars_list, eqs_list, orig_keep_idx

    x0 = x0_full[orig_keep_idx]
    F = compute_jacobian_from_S(S, x0)
    J = (Phi @ F.T).toarray()

    _, R, P = scipy.linalg.qr(J.T, pivoting=True)
    rdiag = np.abs(np.diag(R))
    tol_rank = J.shape[0] * np.finfo(float).eps * rdiag[0] if rdiag[0] > 0 else 0.0
    rank = int(np.sum(rdiag > tol_rank))

    if rank >= n_eqs:
        return S, Phi, vars_list, eqs_list, orig_keep_idx

    keep_eq_rows = np.sort(P[:rank])
    Phi = Phi[keep_eq_rows, :]
    eqs_list = [eqs_list[i] for i in keep_eq_rows]

    return S, Phi, vars_list, eqs_list, orig_keep_idx


def process_cpn_system(problem_ml, jaume_flag=True, zero_derivatives=True) -> CpnSystem:
    """Build and simplify the CPN1 system from a multilinear RMS problem.

    Pipeline:
        1. Build S, Phi matrices, substitute parameters and dx/dt=0
        2. Deduplicate equations
        3. Fixpoint: simplify (var=0) <-> pin constants (var=b) until stable
        4. Handle free variables
        5. Rank reduction: remove linearly dependent equations via QR
        6. Pin variables without trivial monomials (if over-determined)

    Args:
        problem_ml: RmsProblemMultilinear with built multilinear matrices
        jaume_flag: Enable equation dedup, simplification, and rank reduction
        zero_derivatives: Substitute dx/dt=0 for steady-state computation

    Returns:
        CpnSystem with processed S, Phi, variable/equation lists
    """
    variables = problem_ml.state_and_algebraic_vars
    n_sa = len(variables)
    x0 = problem_ml.get_x0()

    vars_list, uid_to_name = build_var_names(problem_ml)
    orig_keep_idx = np.arange(n_sa)

    if zero_derivatives:
        S_vars = problem_ml.S[:n_sa, :]
        diff_rows = problem_ml.S[n_sa:, :]
        deriv_cols = diff_rows.getnnz(axis=0) > 0
        keep_cols = np.where(~deriv_cols)[0]

        S = S_vars[:, keep_cols]
        Phi = problem_ml.Phi[:, keep_cols]
        Phi.eliminate_zeros()

        print(f"  deriv-filter: S={S.shape} Phi={Phi.shape} keep={len(keep_cols)}/{problem_ml.S.shape[1]}  n_sa={n_sa} n_eqs={problem_ml.Phi.shape[0]} n_diff={problem_ml.S.shape[0] - n_sa}")

        subs_map = {dvar: Const(0.0) for dvar in problem_ml._diff_vars}
        if hasattr(problem_ml, '_variable_parameters') and hasattr(problem_ml, '_variable_parameters_values'):
            for p, val in zip(problem_ml._variable_parameters, problem_ml._variable_parameters_values):
                subs_map[p] = Const(float(val))
        if hasattr(problem_ml, '_constant_parameters') and hasattr(problem_ml, '_constant_params'):
            for p, val in zip(problem_ml._constant_parameters, problem_ml._constant_params):
                subs_map[p] = Const(float(val))
        all_raw_eqs = [eq.subs(subs_map).simplify() for eq in problem_ml.get_state_eqs + problem_ml.get_algebraic_eqs]
    else:
        S = problem_ml.S
        Phi = problem_ml.Phi
        # Also include diff_vars in the variable list so S rows match
        diff_vars = list(problem_ml._diff_vars) if hasattr(problem_ml, '_diff_vars') else []
        for dv in diff_vars:
            name = str(dv)
            if name not in vars_list:
                vars_list.append(name)
                uid_to_name[dv.uid] = name
            orig_keep_idx = np.append(orig_keep_idx, -1)
        subs_map = {}
        if hasattr(problem_ml, '_variable_parameters') and hasattr(problem_ml, '_variable_parameters_values'):
            for p, val in zip(problem_ml._variable_parameters, problem_ml._variable_parameters_values):
                subs_map[p] = Const(float(val))
        if hasattr(problem_ml, '_constant_parameters') and hasattr(problem_ml, '_constant_params'):
            for p, val in zip(problem_ml._constant_parameters, problem_ml._constant_params):
                subs_map[p] = Const(float(val))
        all_raw_eqs = [eq.subs(subs_map).simplify() for eq in problem_ml.get_state_eqs + problem_ml.get_algebraic_eqs]

    eqs_list, keep_eq_mask = build_eqs_list(all_raw_eqs, uid_to_name, deduplicate=jaume_flag)

    n_dropped_eqs = np.sum(~keep_eq_mask)
    print(f"  dedup: {len(eqs_list)} eqs from {len(all_raw_eqs)} raw (dropped {n_dropped_eqs} duplicates)")

    pre_const_mask = (np.asarray(S.getnnz(axis=0)).flatten() == 0) & (np.asarray(Phi.getnnz(axis=0)).flatten() > 0)

    if jaume_flag and not np.all(keep_eq_mask):
        keep_eq_idx = np.where(keep_eq_mask)[0]
        Phi = Phi[keep_eq_idx, :]
        S = S[keep_eq_idx, :]
        vars_list = [vars_list[i] for i in keep_eq_idx]
        orig_keep_idx = orig_keep_idx[keep_eq_idx]

        post_const_mask = (np.asarray(S.getnnz(axis=0)).flatten() == 0) & (np.asarray(Phi.getnnz(axis=0)).flatten() > 0)
        phantom = post_const_mask & ~pre_const_mask
        if np.any(phantom):
            keep = np.where(~phantom)[0]
            S = S[:, keep]
            Phi = Phi[:, keep]
        print(f"  dedup-filter: S={S.shape} Phi={Phi.shape}")

    if jaume_flag:
        for iteration in range(100):
            n_eqs_before = Phi.shape[0]
            n_vars_before = S.shape[0]
            n_mon_before = S.shape[1]

            S, Phi, vars_list, eqs_list, orig_keep_idx = simplify_system(
                S, Phi, vars_list, eqs_list, orig_keep_idx
            )

            S, Phi, vars_list, eqs_list, orig_keep_idx = substitute_pinned_constants(
                S, Phi, vars_list, eqs_list, orig_keep_idx
            )

            n_eqs_after = Phi.shape[0]
            n_vars_after = S.shape[0]
            n_mon_after = S.shape[1]

            changed = (n_eqs_before != n_eqs_after or
                       n_vars_before != n_vars_after or
                       n_mon_before != n_mon_after)
            print(f"  fixpoint iter {iteration}: S={n_vars_before}->{n_vars_after} "
                  f"Phi={n_eqs_before}->{n_eqs_after} Mon={n_mon_before}->{n_mon_after}"
                  f"{' [converged]' if not changed else ''}")

            if not changed:
                break

        S, Phi, vars_list, eqs_list, orig_keep_idx = handle_free_variables(
            S, Phi, vars_list, eqs_list, orig_keep_idx, x0
        )
        print(f"  free-vars: S={S.shape} Phi={Phi.shape}")

        S, Phi, vars_list, eqs_list, orig_keep_idx = reduce_rank_qr(
            S, Phi, vars_list, eqs_list, orig_keep_idx, x0
        )
        print(f"  rank-reduce: S={S.shape} Phi={Phi.shape}")

        S, Phi, vars_list, eqs_list, orig_keep_idx = pin_vars_without_trivial_monomials(
            S, Phi, vars_list, eqs_list, orig_keep_idx, x0
        )
        print(f"  pin-orphans: S={S.shape} Phi={Phi.shape}")

    S, Phi, vars_list, eqs_list, orig_keep_idx = cleanup_system(
        S, Phi, vars_list, eqs_list, orig_keep_idx
    )
    print(f"  cleanup: S={S.shape} Phi={Phi.shape}")

    eqs_list = regenerate_eqs_list(S, Phi, vars_list)

    return CpnSystem(S=S, Phi=Phi, vars_list=vars_list, eqs_list=eqs_list,
                     orig_keep_idx=orig_keep_idx, x0=x0)


def remove_numerically_satisfied(S, Phi, vars_list, eqs_list, orig_keep_idx, x0, tol=1e-10):
    """Remove equations whose residual Phi @ varphi(x0) is numerically zero.

    Catches cases the symbolic simplifier misses, e.g.
    ``-(-a*x) - a*x`` which evaluates to 0 but isn't simplified symbolically.

    Returns:
        Updated S, Phi, vars_list, eqs_list, orig_keep_idx
    """
    v = x0[orig_keep_idx]
    n_mon = S.shape[1]
    S_csc = S.tocsc()

    varphi = np.ones(n_mon, dtype=float)
    for j in range(n_mon):
        start = S_csc.indptr[j]
        end = S_csc.indptr[j + 1]
        prod = 1.0
        for p in range(start, end):
            i = S_csc.indices[p]
            s_ij = S_csc.data[p]
            prod *= (s_ij * v[i] + (1.0 - abs(s_ij)))
        varphi[j] = prod

    residual = np.asarray(Phi @ varphi).flatten()
    keep = np.abs(residual) > tol

    Phi = Phi[keep, :]
    eqs_list = [eqs_list[i] for i in np.where(keep)[0]]

    return cleanup_system(S, Phi, vars_list, eqs_list, orig_keep_idx)


def compute_jacobian_from_S(S: sparse.csc_matrix, v: np.ndarray) -> sparse.csc_matrix:
    """Compute the Jacobian of varphi(v) with respect to v.

    For each monomial j: varphi_j(v) = prod_p (s_{i_p,j} * v_{i_p} + (1 - |s_{i_p,j}|))

    The partial derivative d(varphi_j)/d(v_i) = s_ij * prod_{p != i} x_{i_p,j}
    where x_{i_p,j} = s_{i_p,j} * v_{i_p} + (1 - |s_{i_p,j}|)

    Handles the case where x_{i_p,j} = 0 (L'Hopital limit) by computing
    the product of the other factors directly.

    Args:
        S: Variable-to-monomial mapping (n_vars, n_mon)
        v: Current variable values

    Returns:
        F: Jacobian matrix (n_vars, n_mon) where F[i,j] = d(varphi_j)/d(v_i)
    """
    n_vars, n_mon = S.shape
    data = S.data
    indices = S.indices
    indptr = S.indptr

    col_prod = np.ones(n_mon, dtype=float)
    for j in range(n_mon):
        start = indptr[j]
        end = indptr[j + 1]
        prod = 1.0
        for p in range(start, end):
            i = indices[p]
            s_ij = data[p]
            prod *= (s_ij * v[i] + (1.0 - abs(s_ij)))
        col_prod[j] = prod

    f_row: list[int] = []
    f_col: list[int] = []
    f_val: list[float] = []

    for j in range(n_mon):
        start = indptr[j]
        end = indptr[j + 1]
        if start == end:
            continue

        zero_hits = 0
        zero_pos = -1
        for p in range(start, end):
            i = indices[p]
            s_ij = data[p]
            x_ij = s_ij * v[i] + (1.0 - abs(s_ij))
            if abs(x_ij) <= 1e-12:
                zero_hits += 1
                zero_pos = p

        for p in range(start, end):
            i = indices[p]
            s_ij = data[p]
            x_ij = s_ij * v[i] + (1.0 - abs(s_ij))

            if abs(x_ij) > 1e-12:
                val = s_ij * col_prod[j] / x_ij
                f_row.append(i)
                f_col.append(j)
                f_val.append(val)
            elif zero_hits == 1 and p == zero_pos:
                prod_other = 1.0
                for q in range(start, end):
                    if q == p:
                        continue
                    iq = indices[q]
                    s_iq = data[q]
                    x_iq = s_iq * v[iq] + (1.0 - abs(s_iq))
                    prod_other *= x_iq
                val = s_ij * prod_other
                f_row.append(i)
                f_col.append(j)
                f_val.append(val)

    return sparse.csc_matrix((f_val, (f_row, f_col)), shape=(n_vars, n_mon))


def compute_EABC(S: sparse.csc_matrix, Phi: sparse.csr_matrix, v: np.ndarray) -> sparse.csc_matrix:
    """Compute the full system Jacobian J = Phi @ F^T.

    This is the Jacobian of the residual f(v) = Phi @ varphi(v) with respect to v.

    Args:
        S: Variable-to-monomial mapping
        Phi: Equation coefficients
        v: Current variable values

    Returns:
        J: System Jacobian (n_eqs, n_vars)
    """
    F = compute_jacobian_from_S(S, v)
    return (Phi @ F.T).tocsc()
