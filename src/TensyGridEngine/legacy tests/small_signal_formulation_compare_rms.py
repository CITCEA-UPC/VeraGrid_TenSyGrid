# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Compare small-signal routes for non-phasor RMS (RmsProblemDae).

Builds a minimal 2-bus RMS system with the non-phasor generator model and
compares small-signal outputs across raw/explicit/implicit generator block
forms to study route differences (diff_var=0 path vs generalized path).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import VeraGridEngine.api as vge
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from VeraGridEngine.Utils.Symbolic.bus_rms_template import initialize_bus_rms
from VeraGridEngine.Templates.Rms.genqec_exc_gov_sat_template import get_complete_generator_template_rms
from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
from VeraGridEngine.Templates.Rms.load_rms_template import get_load_rms_template
from VeraGridEngine.Simulations.Rms.problems.rms_problem_dae import RmsProblemDae
from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup


def build_problem(formulation: str):
    if formulation not in {"raw", "explicit", "implicit"}:
        raise ValueError(f"Unsupported formulation: {formulation}")

    grid = vge.MultiCircuit(Sbase=100.0, fbase=50.0)
    bus0 = vge.Bus(name="Bus0", Vnom=10, is_slack=True)
    bus1 = vge.Bus(name="Bus1", Vnom=10)
    grid.add_bus(bus0)
    grid.add_bus(bus1)

    for bus in grid.buses:
        initialize_bus_rms(bus, vf=grid.var_factory)

    line = vge.Line(name="Line", bus_from=bus0, bus_to=bus1,
                    r=0.029585798816568046, x=0.07100591715976332, b=0.03, rate=900.0)
    grid.add_line(line)

    load = vge.Load(P=9.999999, Q=0.999999)
    grid.add_load(bus=bus1, api_obj=load)

    gen = vge.Generator(name="Gen0", P=10, vset=1.0, Snom=900)
    grid.add_generator(bus=bus0, api_obj=gen)

    gen_mdl = get_complete_generator_template_rms(grid.var_factory, name="Gen0").block
    if formulation == "explicit":
        gen_mdl = vge.to_explicit(gen_mdl, grid.var_factory)
        _ = 0
    elif formulation == "implicit":
        gen_mdl = vge.to_implicit(gen_mdl, grid.var_factory)

    line_mdl = get_line_rms_template(grid.var_factory).block
    set_rms_model(device=line, model=line_mdl, var_factory=grid.var_factory)
    set_rms_model(device=gen, model=gen_mdl, var_factory=grid.var_factory)

    load_mdl = get_load_rms_template(grid.var_factory).block
    set_rms_model(device=load, model=load_mdl, var_factory=grid.var_factory)

    pf = vge.power_flow(grid, vge.PowerFlowOptions())
    if not pf.converged:
        raise RuntimeError("Power flow failed")

    rms_options = vge.RmsOptions(time_step=0.01, simulation_time=1.0, tolerance=1e-6, max_iter=20)
    problem = RmsProblemDae(grid=grid, options=rms_options, pf_results=pf)
    return problem, pf, rms_options


def run_driver(problem: Any, pf: Any, rms_options: Any):
    problem.set_events_group(RmsEventsGroup("compare_rms"))
    ss_options = vge.RmsSmallSignalStabilityOptions(ss_assessment_time=0)
    driver = vge.SmallSignalStabilityRmsDriver(
        grid=vge.MultiCircuit(Sbase=problem.grid.Sbase),
        rms_options=rms_options,
        sss_options=ss_options,
        pf_results=pf,
    )
    driver.problem = problem
    driver.k = problem.get_states_number()
    driver.run()
    ev = driver.results.eigenvalues
    finite = np.isfinite(ev) & (np.abs(ev) < 1e6)
    finite_ev = ev[finite]
    max_real = float(np.max(np.real(finite_ev))) if len(finite_ev) else float("nan")
    unstable = int(np.sum(np.real(finite_ev) > 1e-6))
    return finite_ev, max_real, unstable


def inspect_augmented_jacobian(problem: Any) -> None:
    """Print diagnostics for the augmented Jacobian used in small signal."""
    x = problem.get_x0()
    dx = np.zeros(problem.get_diff_var_number())
    h = problem.get_dt_value()

    nx = problem.get_states_number()
    ny = problem.get_algebraic_var_number()

    if nx == 0:
        fx = sp.csr_matrix((0, 0))
        fy = sp.csr_matrix((0, ny))
        gx = sp.csr_matrix((ny, 0))
    else:
        fx = problem.get_j11(x, dx, h)
        fy = problem.get_j12(x, dx, h)
        gx = problem.get_j21(x, dx, h)
    gy = problem.get_j22(x, dx, h)

    J_top = sp.hstack([fx, fy])
    J_bot = sp.hstack([gx, gy])
    J_aug = sp.vstack([J_top, J_bot], format="csc")

    print(f"J11 shape={fx.shape} nnz={fx.nnz}")
    print(f"J12 shape={fy.shape} nnz={fy.nnz}")
    print(f"J21 shape={gx.shape} nnz={gx.nnz}")
    print(f"J22 shape={gy.shape} nnz={gy.nnz}")
    print(f"J_aug shape={J_aug.shape} nnz={J_aug.nnz}")

    J_csr = J_aug.tocsr()
    zero_rows = np.flatnonzero(np.diff(J_csr.indptr) == 0)
    zero_cols = np.flatnonzero(np.diff(J_aug.indptr) == 0)
    print(f"zero_rows={len(zero_rows)} zero_cols={len(zero_cols)}")
    if len(zero_rows):
        print(f"  first zero rows: {zero_rows[:20].tolist()}")
    if len(zero_cols):
        print(f"  first zero cols: {zero_cols[:20].tolist()}")

    dense = J_aug.toarray()
    rank = np.linalg.matrix_rank(dense)
    u_svd, svals, vh = la.svd(dense, full_matrices=False)
    smin = float(svals[-1])
    smax = float(svals[0])
    cond = np.inf if smin == 0.0 else smax / smin
    print(f"rank={rank}/{dense.shape[0]} sigma_min={smin:.3e} sigma_max={smax:.3e} cond={cond:.3e}")

    if rank < dense.shape[0]:
        vars_list = list(problem._state_vars) + list(problem._algebraic_vars)
        eqs_list = list(problem._state_eqs) + list(problem._algebraic_eqs)

        right_null = vh[-1, :]
        left_null = u_svd[:, -1]

        print("Approximate right-null direction (variables, top 12):")
        ridx = np.argsort(np.abs(right_null))[::-1][:12]
        for idx in ridx:
            if abs(right_null[idx]) < 1e-8:
                continue
            vname = str(vars_list[idx]) if idx < len(vars_list) else f"var[{idx}]"
            print(f"  col={idx:<4d} {vname:<40s} coeff={right_null[idx]:+.4e}")

        print("Approximate left-null direction (equations, top 12):")
        lidx = np.argsort(np.abs(left_null))[::-1][:12]
        for idx in lidx:
            if abs(left_null[idx]) < 1e-8:
                continue
            ename = str(eqs_list[idx]) if idx < len(eqs_list) else f"eq[{idx}]"
            print(f"  row={idx:<4d} coeff={left_null[idx]:+.4e} eq={ename}")

    E = problem.get_E_matrix(x, dx)
    E_dense = E.toarray() if hasattr(E, "toarray") else np.array(E)
    Erank = np.linalg.matrix_rank(E_dense)
    Es = la.svdvals(E_dense)
    Esmin = float(Es[-1])
    Esmax = float(Es[0])
    Econd = np.inf if Esmin == 0.0 else Esmax / Esmin
    print(f"E shape={E_dense.shape} rank={Erank}/{E_dense.shape[0]} sigma_min={Esmin:.3e} cond={Econd:.3e}")


def main() -> int:
    for formulation in ["raw", "explicit", "implicit"]:
        print("\n" + "=" * 80)
        print(f"RMS FORMULATION: {formulation}")
        print("=" * 80)
        try:
            problem, pf, rms_options = build_problem(formulation)
            finite_ev, max_real, unstable = run_driver(problem, pf, rms_options)
            print(
                f"states={problem.get_states_number()} diff_vars={problem.get_diff_var_number()} "
                f"alg_vars={problem.get_algebraic_var_number()}"
            )
            print(f"finite_eigs={len(finite_ev)} unstable={unstable} max_real={max_real:+.6f}")
            print("finite eigenvalues:")
            for i, ev in enumerate(finite_ev, start=1):
                print(f"  mode={i:<3d} ev={np.real(ev):+.6f}{np.imag(ev):+.6f}j")
        except Exception as exc:
            print(f"FAILED: {exc}")
            try:
                print("-- Jacobian diagnostics after failure --")
                inspect_augmented_jacobian(problem)
            except Exception as sub_exc:
                print(f"diagnostics_failed: {sub_exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
