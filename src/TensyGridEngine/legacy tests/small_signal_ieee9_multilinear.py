# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

project_base = next(p for p in Path(__file__).resolve().parents if (p / "src" / "VeraGridEngine").exists())
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import VeraGridEngine.api as vge
from VeraGridEngine.Simulations.Rms.numerical.back_euler_fx import BackEulerImplicitIntegration
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model


def ensure_unique_device_names(grid) -> None:
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


def sanitize_runtime_parameter_names(grid) -> None:
    def _sanitize_block(block, tag: str) -> None:
        block.unify_blocks()
        all_syms = list(block.algebraic_vars) + list(block.state_vars) + list(block.diff_vars)
        all_syms += list(block.event_dict.keys()) + list(block.mode_dict.keys())
        seen: dict[str, int] = {}
        for k, sym in enumerate(all_syms):
            name = str(sym.name)
            if name not in seen:
                seen[name] = 0
                continue
            seen[name] += 1
            sym.name = f"{name}__{tag}_{seen[name]}_{k}"

    for i, dev in enumerate(grid.get_branches_iter(add_vsc=True, add_hvdc=True, add_switch=True)):
        if not dev.rms_model.empty():
            _sanitize_block(dev.rms_model, f"br{i}")

    for i, dev in enumerate(grid.get_injection_devices_iter()):
        if not dev.rms_model.empty():
            _sanitize_block(dev.rms_model, f"inj{i}")


def _compare_eigen_sets(ref_eigs: np.ndarray, test_eigs: np.ndarray, tol: float = 1e-6) -> tuple[int, int, int, float]:
    if len(ref_eigs) == 0 or len(test_eigs) == 0:
        return 0, len(ref_eigs), len(test_eigs), float("nan")

    candidates: list[tuple[float, int, int]] = []
    for i, ev in enumerate(ref_eigs):
        d = np.abs(test_eigs - ev)
        valid = np.where(d <= tol)[0]
        for j in valid:
            candidates.append((float(d[j]), i, int(j)))

    candidates.sort(key=lambda t: t[0])

    used_ref: set[int] = set()
    used_test: set[int] = set()
    errs: list[float] = []
    for err, i, j in candidates:
        if i in used_ref or j in used_test:
            continue
        used_ref.add(i)
        used_test.add(j)
        errs.append(err)

    matches = len(errs)

    mean_err = float(np.mean(errs)) if errs else float("nan")
    return matches, len(ref_eigs) - matches, len(test_eigs) - matches, mean_err


def _compute_unmatched(ref_eigs: np.ndarray, test_eigs: np.ndarray, tol: float = 1e-6) -> tuple[list[tuple[complex, float]], list[tuple[complex, float]]]:
    if len(ref_eigs) == 0 or len(test_eigs) == 0:
        return [], []

    candidates: list[tuple[float, int, int]] = []
    for i, ev in enumerate(ref_eigs):
        d = np.abs(test_eigs - ev)
        for j, dist in enumerate(d):
            if dist <= tol:
                candidates.append((float(dist), i, j))

    candidates.sort(key=lambda t: t[0])
    used_ref: set[int] = set()
    used_test: set[int] = set()
    for _, i, j in candidates:
        if i in used_ref or j in used_test:
            continue
        used_ref.add(i)
        used_test.add(j)

    ref_unmatched: list[tuple[complex, float]] = []
    for i, ev in enumerate(ref_eigs):
        if i in used_ref:
            continue
        nearest = float(np.min(np.abs(test_eigs - ev)))
        ref_unmatched.append((complex(ev), nearest))

    test_unmatched: list[tuple[complex, float]] = []
    for j, ev in enumerate(test_eigs):
        if j in used_test:
            continue
        nearest = float(np.min(np.abs(ref_eigs - ev)))
        test_unmatched.append((complex(ev), nearest))

    ref_unmatched.sort(key=lambda x: x[1])
    test_unmatched.sort(key=lambda x: x[1])
    return ref_unmatched, test_unmatched


def _print_unmatched(ref_unmatched: list[tuple[complex, float]],
                     test_unmatched: list[tuple[complex, float]],
                     n_show: int = 8) -> None:
    print("Unmatched modes (nearest-neighbor distance)")
    print(f"  ML-only (showing up to {n_show})")
    for ev, dist in ref_unmatched[:n_show]:
        print(f"    ev={ev.real:+.9e}{ev.imag:+.9e}j  nearest={dist:.3e}")

    print(f"  Phasor-only (showing up to {n_show})")
    for ev, dist in test_unmatched[:n_show]:
        print(f"    ev={ev.real:+.9e}{ev.imag:+.9e}j  nearest={dist:.3e}")


def _plot_eigen_comparison(ml_eigs: np.ndarray, ph_eigs: np.ndarray, grid_label: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(np.real(ml_eigs), np.imag(ml_eigs), s=40, alpha=0.7, c="tab:blue", label="ML driver")
    ax.scatter(np.real(ph_eigs), np.imag(ph_eigs), s=40, alpha=0.7, c="tab:orange", marker="x", label="Phasor driver")
    ax.axvline(x=0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(f"{grid_label} Eigenvalue Map (ML vs Phasor)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plot_dir = Path(__file__).resolve().parent / "plots"
    plot_dir.mkdir(exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in grid_label).strip("_")
    plot_path = plot_dir / f"{safe_label}_eigen_comparison.png"
    fig.savefig(plot_path, dpi=160)
    print(f"Saved eigen comparison plot: {plot_path}")
    plt.show()
    plt.close(fig)


def _seed_problem_init_guess_from_vector(problem, x_vec: np.ndarray) -> None:
    for uid, idx in problem.uid2idx_vars.items():
        if idx < len(x_vec):
            problem.init_guess[uid] = float(x_vec[idx])


def _debug_kcl_uid_trace(problem, marker: str = "Irg_Gen9") -> None:
    eqs_attr = getattr(problem, "get_algebraic_eqs", None)
    eqs = eqs_attr() if callable(eqs_attr) else eqs_attr
    if eqs is None:
        eqs = getattr(problem, "_algebraic_eqs", [])
    x0 = problem.get_x0()
    print(f"\nKCL UID trace (marker={marker})")
    for i, eq in enumerate(eqs):
        eq_text = str(eq)
        if marker not in eq_text:
            continue
        print(f"eq_idx={i} eq={eq_text}")
        vars_in_eq = eq.get_vars()
        shown = set()
        for v in vars_in_eq:
            if v.uid in shown:
                continue
            shown.add(v.uid)
            idx = problem.uid2idx_vars.get(v.uid, None)
            x0_val = float(x0[idx]) if idx is not None and idx < len(x0) else float("nan")
            init_val = problem.init_guess.get(v.uid, None)
            print(f"  var={v.name} uid={v.uid} idx={idx} x0={x0_val:+.9e} init={init_val}")


def _debug_bus38_kcl_from_pf_seed(problem) -> None:
    state_vars_attr = getattr(problem, "get_state_vars", None)
    alg_vars_attr = getattr(problem, "get_algebraic_vars", None)
    state_vars = state_vars_attr() if callable(state_vars_attr) else getattr(problem, "_state_vars", [])
    alg_vars = alg_vars_attr() if callable(alg_vars_attr) else getattr(problem, "_algebraic_vars", [])
    all_vars = list(state_vars) + list(alg_vars)
    name_to_uid = {str(v.name): v.uid for v in all_vars}
    x0 = problem.get_x0()

    def _lookup_uid_by_prefix(prefix: str) -> int:
        for n, uid in name_to_uid.items():
            if n == prefix or n.startswith(prefix + "__"):
                return uid
        raise KeyError(prefix)

    def val(name: str) -> float:
        uid = _lookup_uid_by_prefix(name)
        idx = problem.uid2idx_vars[uid]
        return float(x0[idx])

    irt1 = val("Irt_branch 1")
    irt16 = val("Irt_branch 16")
    irf1 = val("Irf_branch 1")
    irf16 = val("Irf_branch 16")
    iit1 = val("Iit_branch 1")
    iit16 = val("Iit_branch 16")
    iif1 = val("Iif_branch 1")
    iif16 = val("Iif_branch 16")
    irg = val("Irg_Gen9")
    iig = val("Iig_Gen9")
    irl = val("Ir_Load1@bus 38")
    iil = val("Ii_Load1@bus 38")

    kcl_r_using_t = -irt1 - irt16 + irg + irl
    kcl_i_using_t = -iit1 - iit16 + iig + iil
    kcl_r_using_f = -irf1 - irf16 + irg + irl
    kcl_i_using_f = -iif1 - iif16 + iig + iil

    print("\nBus 38 KCL check from PF-seeded x0")
    print(f"  using to-side vars : R={kcl_r_using_t:+.9e} I={kcl_i_using_t:+.9e}")
    print(f"  using from-side vars: R={kcl_r_using_f:+.9e} I={kcl_i_using_f:+.9e}")
    print("  terms (to-side):")
    print(f"    Irt1={irt1:+.9e} Irt16={irt16:+.9e} Irg9={irg:+.9e} IrLoad38={irl:+.9e}")
    print(f"    Iit1={iit1:+.9e} Iit16={iit16:+.9e} Iig9={iig:+.9e} IiLoad38={iil:+.9e}")


def _debug_bus38_power_balance(problem) -> None:
    eqs_attr = getattr(problem, "get_algebraic_eqs", None)
    eqs = eqs_attr() if callable(eqs_attr) else eqs_attr
    if eqs is None:
        eqs = getattr(problem, "_algebraic_eqs", [])
    x0 = problem.get_x0()

    def get_val_by_uid(uid: int) -> float:
        idx = problem.uid2idx_vars[uid]
        return float(x0[idx])

    eq_r = None
    eq_i = None
    for eq in eqs:
        txt = str(eq)
        if "Irg_Gen9" in txt and "Ir_Load1@bus 38" in txt and "Irt_branch" in txt:
            eq_r = eq
        if "Iig_Gen9" in txt and "Ii_Load1@bus 38" in txt and "Iit_branch" in txt:
            eq_i = eq

    if eq_r is None or eq_i is None:
        print("\nBus 38 power-balance debug skipped: could not locate KCL equations")
        return

    vars_r = {str(v.name): get_val_by_uid(v.uid) for v in eq_r.get_vars()}
    vars_i = {str(v.name): get_val_by_uid(v.uid) for v in eq_i.get_vars()}

    vr = None
    vi = None
    state_vars = getattr(problem, "_state_vars", [])
    alg_vars = getattr(problem, "_algebraic_vars", [])
    for v in list(state_vars) + list(alg_vars):
        n = str(v.name)
        if n == "Vr":
            vr = get_val_by_uid(v.uid)
        if n == "Vi":
            vi = get_val_by_uid(v.uid)
    if vr is None or vi is None:
        print("\nBus 38 power-balance debug skipped: could not find Vr/Vi")
        return

    ir_mis = -vars_r["Irt_branch 1"] - vars_r["Irt_branch 16"] + vars_r["Irg_Gen9"] + vars_r["Ir_Load1@bus 38"]
    ii_mis = -vars_i["Iit_branch 1"] - vars_i["Iit_branch 16"] + vars_i["Iig_Gen9"] + vars_i["Ii_Load1@bus 38"]

    v = complex(vr, vi)
    i_mis = complex(ir_mis, ii_mis)
    s_mis = v * np.conj(i_mis)

    print("\nBus 38 power-balance check from KCL mismatch")
    print(f"  V38={vr:+.9e}{vi:+.9e}j")
    print(f"  I_mismatch={ir_mis:+.9e}{ii_mis:+.9e}j")
    print(f"  S_mismatch=V*conj(I_mismatch)={s_mis.real:+.9e}{s_mis.imag:+.9e}j")


def _debug_gen9_subexciter_values(problem) -> None:
    x0 = problem.get_x0()
    state_vars_attr = getattr(problem, "get_state_vars", None)
    alg_vars_attr = getattr(problem, "get_algebraic_vars", None)
    state_vars = state_vars_attr() if callable(state_vars_attr) else getattr(problem, "_state_vars", [])
    alg_vars = alg_vars_attr() if callable(alg_vars_attr) else getattr(problem, "_algebraic_vars", [])

    print("\nGen9 subexciter debug values")
    found = 0
    for var in list(state_vars) + list(alg_vars):
        name = str(var.name)
        low = name.lower()
        if "gen9" not in low:
            continue
        if "subexciter" not in low and "exciter" not in low and "kc" not in low:
            continue
        idx = problem.uid2idx_vars.get(var.uid, None)
        x0_val = float(x0[idx]) if idx is not None and idx < len(x0) else float("nan")
        init_val = problem.init_guess.get(var.uid, None)
        print(f"  name={name} uid={var.uid} idx={idx} x0={x0_val:+.9e} init={init_val}")
        found += 1

    if found == 0:
        print("  No Gen9 subexciter/exciter vars found with current name filters")


def _debug_generator_exciter_all_vars(problem, gen_index: int = 9) -> None:
    grid = getattr(problem, "grid", None)
    if grid is None or not hasattr(grid, "generators"):
        print(f"\nGen{gen_index} exciter full dump skipped: problem.grid.generators not available")
        return

    target = None
    for gen in grid.generators:
        gname = str(getattr(gen, "name", "")).strip().lower().replace(" ", "")
        if gname == f"gen{gen_index}":
            target = gen
            break
    if target is None and len(grid.generators) > gen_index:
        target = grid.generators[gen_index]

    if target is None:
        print(f"\nGen{gen_index} exciter full dump skipped: target generator not found")
        return

    mdl = getattr(target, "rms_model", None)
    if mdl is None or mdl.empty():
        print(f"\nGen{gen_index} exciter full dump skipped: target RMS model empty")
        return

    x0 = problem.get_x0()
    seen: set[int] = set()

    def _collect(block, attr: str):
        vals = getattr(block, attr, [])
        return list(vals) if vals is not None else []

    vars_all = []
    vars_all += _collect(mdl, "state_vars")
    vars_all += _collect(mdl, "algebraic_vars")
    vars_all += _collect(mdl, "diff_vars")
    vars_all += _collect(mdl, "in_vars")
    vars_all += _collect(mdl, "out_vars")

    print(f"\nGen{gen_index} exciter/full-generator variable dump")
    print(f"  generator_name={getattr(target, 'name', 'unknown')} total_candidates={len(vars_all)}")
    for var in vars_all:
        uid = getattr(var, "uid", None)
        if uid is None or uid in seen:
            continue
        seen.add(uid)
        name = str(getattr(var, "name", ""))
        x_idx = problem.uid2idx_vars.get(uid, None)
        x0_val = float(x0[x_idx]) if x_idx is not None and x_idx < len(x0) else float("nan")
        init_val = problem.init_guess.get(uid, None)
        print(f"  name={name} uid={uid} x_idx={x_idx} x0={x0_val:+.9e} init={init_val}")


def _debug_vars_by_compiler_index(problem, indices: list[int]) -> None:
    x0 = problem.get_x0()
    state_vars_attr = getattr(problem, "get_state_vars", None)
    alg_vars_attr = getattr(problem, "get_algebraic_vars", None)
    state_vars = state_vars_attr() if callable(state_vars_attr) else getattr(problem, "_state_vars", [])
    alg_vars = alg_vars_attr() if callable(alg_vars_attr) else getattr(problem, "_algebraic_vars", [])
    all_vars = list(state_vars) + list(alg_vars)

    print("\nCompiler-index variable trace")
    for idx in indices:
        if idx < 0 or idx >= len(all_vars):
            print(f"  idx={idx} out-of-range (n_vars={len(all_vars)})")
            continue
        var = all_vars[idx]
        uid = var.uid
        x_idx = problem.uid2idx_vars.get(uid, None)
        x0_val = float(x0[x_idx]) if x_idx is not None and x_idx < len(x0) else float("nan")
        init_val = problem.init_guess.get(uid, None)
        print(f"  idx={idx} name={var.name} uid={uid} x_idx={x_idx} x0={x0_val:+.9e} init={init_val}")
    _=0


def _print_participation_summary(problem, eigenvalues: np.ndarray, participation_factors: np.ndarray, label: str, n_show: int = 8) -> None:
    if participation_factors is None or participation_factors.size == 0:
        print(f"{label} participation factors: unavailable")
        return
    state_vars = list(getattr(problem, "_state_vars", []))
    alg_vars = list(getattr(problem, "_algebraic_vars", []))
    all_dyn_vars = state_vars + alg_vars
    finite_idx = np.where(np.isfinite(eigenvalues) & (np.abs(eigenvalues) < 1e6))[0]
    unstable = [i for i in finite_idx if float(np.real(eigenvalues[i])) > 0.0]
    stable_or_marginal = [i for i in finite_idx if float(np.real(eigenvalues[i])) <= 0.0]
    ordered_idx = unstable + stable_or_marginal
    print(f"{label} dominant participation (top {min(n_show, len(finite_idx))} finite modes)")

    def _var_name(k: int) -> str:
        if k < len(state_vars):
            return f"state:{state_vars[k]}"
        if k < len(all_dyn_vars):
            return f"alg:{all_dyn_vars[k]}"
        return f"var[{k}]"

    for i in ordered_idx[:n_show]:
        if i >= participation_factors.shape[1] or participation_factors.shape[0] == 0:
            continue
        pf = np.abs(participation_factors[:, i])
        k = int(np.argmax(pf))
        name = _var_name(k)
        status = "UNSTABLE" if float(np.real(eigenvalues[i])) > 0.0 else "stable/marginal"
        print(f"  mode={i + 1} ev={eigenvalues[i].real:+.6e}{eigenvalues[i].imag:+.6e}j {status} dominant={name} PF={pf[k]:.3e}")

    if len(unstable) == 0:
        return

    print(f"{label} unstable modes: top 5 dominant states")
    for i in unstable:
        if i >= participation_factors.shape[1] or participation_factors.shape[0] == 0:
            continue
        pf = np.abs(participation_factors[:, i])
        top_idx = np.argsort(pf)[::-1][:5]
        dom = ", ".join([f"{_var_name(int(k))}={pf[int(k)]:.3e}" for k in top_idx])
        print(f"  mode={i + 1} ev={eigenvalues[i].real:+.6e}{eigenvalues[i].imag:+.6e}j -> {dom}")


def _scale_event_dict_var(block, var_name: str, scale: float) -> int:
    count = 0
    for var, value in block.event_dict.items():
        if var.name == var_name:
            value.value = float(value.value) * scale
            count += 1
    for child in block.children:
        count += _scale_event_dict_var(child, var_name, scale)
    return count


def _tune_generator_controller(block, ks_scale: float = 1.0, ka_scale: float = 1.0, ta_scale: float = 1.0) -> dict[str, int]:
    updates = {"Ks": 0, "Ka": 0, "tA": 0}
    if abs(ks_scale - 1.0) > 1e-12:
        updates["Ks"] = _scale_event_dict_var(block, "Ks", ks_scale)
    if abs(ka_scale - 1.0) > 1e-12:
        updates["Ka"] = _scale_event_dict_var(block, "Ka", ka_scale)
    if abs(ta_scale - 1.0) > 1e-12:
        updates["tA"] = _scale_event_dict_var(block, "tA", ta_scale)
    return updates


def _scale_stabilizer_gain(grid, scale: float) -> int:
    updated = 0
    for gen in grid.generators:
        if gen.rms_model.empty():
            continue
        updated += _scale_event_dict_var(gen.rms_model, "Ks", scale)
    return updated


def _apply_bus38_kcl_q_correction(problem) -> None:
    eqs_attr = getattr(problem, "get_algebraic_eqs", None)
    eqs = eqs_attr() if callable(eqs_attr) else eqs_attr
    if eqs is None:
        eqs = getattr(problem, "_algebraic_eqs", [])
    x0 = problem.get_x0()

    def get_val_by_uid(uid: int) -> float:
        idx = problem.uid2idx_vars[uid]
        return float(x0[idx])

    eq_r = None
    eq_i = None
    for eq in eqs:
        txt = str(eq)
        if "Irg_Gen9" in txt and "Ir_Load1@bus 38" in txt and "Irt_branch" in txt:
            eq_r = eq
        if "Iig_Gen9" in txt and "Ii_Load1@bus 38" in txt and "Iit_branch" in txt:
            eq_i = eq

    if eq_r is None or eq_i is None:
        print("Bus 38 KCL correction skipped: equations not found")
        return

    vars_r = {str(v.name): v for v in eq_r.get_vars()}
    vars_i = {str(v.name): v for v in eq_i.get_vars()}

    ir_mis = (
        -get_val_by_uid(vars_r["Irt_branch 1"].uid)
        -get_val_by_uid(vars_r["Irt_branch 16"].uid)
        +get_val_by_uid(vars_r["Irg_Gen9"].uid)
        +get_val_by_uid(vars_r["Ir_Load1@bus 38"].uid)
    )
    ii_mis = (
        -get_val_by_uid(vars_i["Iit_branch 1"].uid)
        -get_val_by_uid(vars_i["Iit_branch 16"].uid)
        +get_val_by_uid(vars_i["Iig_Gen9"].uid)
        +get_val_by_uid(vars_i["Ii_Load1@bus 38"].uid)
    )

    ir_load_var = vars_r["Ir_Load1@bus 38"]
    ii_load_var = vars_i["Ii_Load1@bus 38"]
    ir_old = problem.init_guess.get(ir_load_var.uid, get_val_by_uid(ir_load_var.uid))
    ii_old = problem.init_guess.get(ii_load_var.uid, get_val_by_uid(ii_load_var.uid))

    # Apply current correction from KCL mismatch (this corresponds to pure Q correction at bus 38 in this case)
    problem.init_guess[ir_load_var.uid] = float(ir_old - ir_mis)
    problem.init_guess[ii_load_var.uid] = float(ii_old - ii_mis)
    print("Applied bus 38 KCL/Q correction to load current init:")
    print(f"  Ir_Load1@bus 38: {ir_old:+.9e} -> {problem.init_guess[ir_load_var.uid]:+.9e}")
    print(f"  Ii_Load1@bus 38: {ii_old:+.9e} -> {problem.init_guess[ii_load_var.uid]:+.9e}")


def build_problems(
    grid_filename: str,
    pss_gain_scale: float = 1.0,
    gen0_ks_scale: float = 1.0,
    gen0_ka_scale: float = 1.0,
    gen0_ta_scale: float = 1.0,
) -> tuple[vge.RmsProblemMultilinear, vge.RmsProblemPhasor, vge.PowerFlowResults, vge.RmsOptions, vge.RmsOptions]:
    grid_path = project_base / "Grids_and_profiles" / "grids" / grid_filename
    grid = vge.open_file(str(grid_path))
    ensure_unique_device_names(grid)
    sanitize_runtime_parameter_names(grid)

    for load in grid.loads:
        if getattr(load, "bus", None) is not None and getattr(load.bus, "name", "") == "bus 38":
            if hasattr(load, "Q"):
                print(f"Temporarily zeroing Q at {load.bus.name}: Q={load.Q}")
                load.Q = 0.0

    for bus in grid.buses:
        if bus.rms_model.empty():
            vge.initialize_bus_phasor_rms(bus, vf=grid.var_factory)

    for igen, gen in enumerate(grid.generators):
        if not gen.rms_model.empty():
            continue
        gen_mdl = vge.get_complete_generator_template_phasor(grid.var_factory, name=f"Gen{igen}").block
        if igen == 0:
            updates = _tune_generator_controller(
                gen_mdl,
                ks_scale=gen0_ks_scale,
                ka_scale=gen0_ka_scale,
                ta_scale=gen0_ta_scale,
            )
            expected_updates = {
                "Ks": abs(gen0_ks_scale - 1.0) > 1e-12,
                "Ka": abs(gen0_ka_scale - 1.0) > 1e-12,
                "tA": abs(gen0_ta_scale - 1.0) > 1e-12,
            }
            missing = [name for name, expected in expected_updates.items() if expected and updates[name] == 0]
            if missing:
                raise RuntimeError(f"Gen0 tuning requested but event vars were not found: {missing}")
            print(
                f"Gen0 tuning: Ks x{gen0_ks_scale:.3f}, Ka x{gen0_ka_scale:.3f}, tA x{gen0_ta_scale:.3f}"
                f" -> updates {updates}"
            )
        grid.var_factory.add_connections([gen_mdl.in_vars[0]], [gen.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([gen_mdl.in_vars[1]], [gen.bus.rms_model.out_vars[1]])
        gen_mdl = vge.to_implicit(gen_mdl, grid.var_factory)
        set_rms_model(device=gen, model=gen_mdl, var_factory=grid.var_factory)

    for line in grid.lines:
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
        if not load.rms_model.empty():
            continue
        load_mdl = vge.get_load_phasor_current_rms_template(grid.var_factory, name=load.name).block
        grid.var_factory.add_connections([load_mdl.in_vars[0]], [load.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([load_mdl.in_vars[1]], [load.bus.rms_model.out_vars[1]])
        set_rms_model(device=load, model=load_mdl, var_factory=grid.var_factory)

    for trafo in grid.transformers2w:
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
        if not shunt.rms_model.empty():
            continue
        shunt_mdl = vge.get_shunt_template(grid.var_factory, name=shunt.name, phasor=True).block
        grid.var_factory.add_connections([shunt_mdl.in_vars[0]], [shunt.bus.rms_model.out_vars[0]])
        grid.var_factory.add_connections([shunt_mdl.in_vars[1]], [shunt.bus.rms_model.out_vars[1]])
        shunt_mdl = vge.to_implicit(shunt_mdl, grid.var_factory)
        set_rms_model(device=shunt, model=shunt_mdl, var_factory=grid.var_factory)

    if abs(pss_gain_scale - 1.0) > 1e-12:
        n_scaled = _scale_stabilizer_gain(grid, pss_gain_scale)
        print(f"Applied PSS gain scale={pss_gain_scale:.3f} to Ks entries: {n_scaled}")

    pf_results = vge.power_flow(grid, vge.PowerFlowOptions(tolerance=1e-5))
    if not pf_results.converged:
        raise RuntimeError("Power flow did not converge")

    rms_options_ml = vge.RmsOptions(
        time_step=0.01,
        simulation_time=1.0,
        tolerance=1e-6,
        max_iter=20,
        problem_type=vge.RmsProblemTypes.Multilinear,
    )
    rms_options_ph = vge.RmsOptions(
        time_step=0.01,
        simulation_time=1.0,
        tolerance=1e-6,
        max_iter=20,
    )
    problem_ml = vge.RmsProblemMultilinear(grid=grid, options=rms_options_ml, pf_results=pf_results)
    problem_ph = vge.RmsProblemPhasor(grid=grid, options=rms_options_ph, pf_results=pf_results)
    return problem_ml, problem_ph, pf_results, rms_options_ml, rms_options_ph


def run_small_signal_from_driver(
    problem,
    pf_results: vge.PowerFlowResults,
    rms_options: vge.RmsOptions,
    k: int | None = None,
    verbose: int = 0,
):
    ss_options = vge.RmsSmallSignalStabilityOptions(ss_assessment_time=0, verbose=verbose)
    n_modes = problem.get_states_number() + problem.get_diff_var_number()
    k_eff = int(n_modes*0.5)
    ss_options.k = k_eff
    driver = vge.SmallSignalStabilityRmsDriver(
        grid=vge.MultiCircuit(Sbase=problem.grid.Sbase),
        rms_options=rms_options,
        sss_options=ss_options,
        pf_results=pf_results,
    )
    driver.problem = problem
    driver.k = k_eff
    driver.run()
    return driver.results.eigenvalues, driver.results.participation_factors

def run_cpn_tests(problem_ml: vge.RmsProblemMultilinear) -> None:
    rank = 150
    max_iter = 20
    tol = 1e-3

    profiler = cProfile.Profile()
    profiler.enable()
    cp_tensor = problem_ml.fit_cptensor_from_s_phi(
        rank=rank,
        max_iter=max_iter,
        tol=tol,
        random_state=0,
        verbose=True,
    )
    profiler.disable()

    out_dir = Path(__file__).resolve().parent
    profile_pstats_path = out_dir / "cpn_fit_profile.pstats"
    profile_txt_path = out_dir / "cpn_fit_profile.txt"

    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(30)
    stats.dump_stats(str(profile_pstats_path))
    profile_txt_path.write_text(stats_stream.getvalue(), encoding="utf-8")

    print("CPN fit profiler (top 30 cumulative):")
    print(stats_stream.getvalue())
    print(f"Saved profiler stats: {profile_pstats_path}")
    print(f"Saved profiler text : {profile_txt_path}")

    approximator = getattr(problem_ml, "_last_cpn_approximator", None)
    diagnostics = getattr(approximator, "diagnostics", None)
    if diagnostics is not None:
        print("CPN fit diagnostics:")
        print(f"  rank={rank} max_iter={max_iter} tol={tol:.1e}")
        print(f"  converged={diagnostics.converged} iterations={diagnostics.iterations}")
        if diagnostics.relative_changes:
            print(f"  final_relative_change={diagnostics.relative_changes[-1]:.6e}")

    x = problem_ml.get_x0()
    dx = np.zeros(problem_ml.get_diff_var_number())
    x_eval = np.concatenate([x, dx])
    f_exact = problem_ml.exact_residual_from_s_phi(x=x_eval)
    f_cpn = problem_ml.evaluate_residual_from_cptensor(x=x_eval, cp_tensor=cp_tensor)
    err = f_cpn - f_exact
    abs_err = float(np.linalg.norm(err))
    rel_err = abs_err / max(float(np.linalg.norm(f_exact)), 1e-12)

    print("CPN test:")
    print(f"  exact_norm={np.linalg.norm(f_exact):.6e}")
    print(f"  cpn_norm={np.linalg.norm(f_cpn):.6e}")
    print(f"  abs_err={abs_err:.6e}")
    print(f"  rel_err={rel_err:.6e}")

    rng = np.random.default_rng(0)
    scales = [1e-4, 1e-3, 1e-2]
    print("CPN random-x checks:")
    for scale in scales:
        x_rand = x_eval + scale * rng.standard_normal(x_eval.shape[0])
        f_exact_rand = problem_ml.exact_residual_from_s_phi(x=x_rand)
        f_cpn_rand = problem_ml.evaluate_residual_from_cptensor(x=x_rand, cp_tensor=cp_tensor)
        err_rand = f_cpn_rand - f_exact_rand
        abs_err_rand = float(np.linalg.norm(err_rand))
        rel_err_rand = abs_err_rand / max(float(np.linalg.norm(f_exact_rand)), 1e-12)
        print(f"  scale={scale:.0e} exact_norm={np.linalg.norm(f_exact_rand):.6e} cpn_norm={np.linalg.norm(f_cpn_rand):.6e} abs_err={abs_err_rand:.6e} rel_err={rel_err_rand:.6e}")
    return

def main() -> None:
    grid_filename = sys.argv[1] if len(sys.argv) > 1 else "IEEE39.gridcal"
    gen0_tuning_cases = [
        (1.0, 1.0, 1.0),
        (0.7, 0.7, 2.0),
        (0.5, 0.5, 3.0),
        (0.3, 0.4, 5.0),
        (0.1, 0.25, 8.0),
    ]
    for gen0_ks_scale, gen0_ka_scale, gen0_ta_scale in gen0_tuning_cases:
        print(
            f"\n=== Gen0 controller tuning: Ks x{gen0_ks_scale:.3f}, "
            f"Ka x{gen0_ka_scale:.3f}, tA x{gen0_ta_scale:.3f} ==="
        )
        problem_ml, problem_ph, pf_results, rms_options_ml, rms_options_ph = build_problems(
            grid_filename=grid_filename,
            gen0_ks_scale=gen0_ks_scale,
            gen0_ka_scale=gen0_ka_scale,
            gen0_ta_scale=gen0_ta_scale,
        )

        print("Running phasor warm-start micro-step...")
        # Keep the sweep readable; enable these only for detailed initialization debugging.
        # _debug_gen9_subexciter_values(problem_ph)
        # _debug_generator_exciter_all_vars(problem_ph, gen_index=0)
        # _debug_vars_by_compiler_index(problem_ph, indices=[1402, 1461])
        # _debug_kcl_uid_trace(problem_ph, marker="Irg_Gen9")
        # _debug_bus38_power_balance(problem_ph)
        _apply_bus38_kcl_q_correction(problem_ph)
        problem_ph.reset_boundary_update_state(0.0)
        warm_solver = BackEulerImplicitIntegration(
            problem=problem_ph,
            t0=0.0,
            t_end=2e-4,
            h=1e-4,
            max_iter=300,
        )
        _, y_warm, warm_well_initialized, warm_converged = warm_solver.simulate()
        print(f"  warm_converged={bool(warm_converged)} warm_well_initialized={bool(warm_well_initialized)}")
        if warm_converged and y_warm is not None and y_warm.shape[0] > 0:
            _seed_problem_init_guess_from_vector(problem_ph, y_warm[-1, :])

        eig_driver_ml, pf_driver_ml = run_small_signal_from_driver(problem=problem_ml,
                                                                    pf_results=pf_results,
                                                                    rms_options=rms_options_ml)
        eig_driver_ph, pf_driver_ph = run_small_signal_from_driver(problem=problem_ph,
                                                                    pf_results=pf_results,
                                                                    rms_options=rms_options_ph)

        fin_drv_ml = eig_driver_ml[np.isfinite(eig_driver_ml) & (np.abs(eig_driver_ml) < 1e6)]
        fin_drv_ph = eig_driver_ph[np.isfinite(eig_driver_ph) & (np.abs(eig_driver_ph) < 1e6)]

        stable_ml = bool(np.all(np.real(fin_drv_ml) <= 0.0)) if len(fin_drv_ml) > 0 else False
        margin_ml = float(np.max(np.real(fin_drv_ml))) if len(fin_drv_ml) > 0 else float("nan")
        stable_ph = bool(np.all(np.real(fin_drv_ph) <= 0.0)) if len(fin_drv_ph) > 0 else False
        margin_ph = float(np.max(np.real(fin_drv_ph))) if len(fin_drv_ph) > 0 else float("nan")
        n_unstable_ml = int(np.sum(np.real(fin_drv_ml) > 0.0)) if len(fin_drv_ml) > 0 else 0
        n_unstable_ph = int(np.sum(np.real(fin_drv_ph) > 0.0)) if len(fin_drv_ph) > 0 else 0

        matches, ml_only, ph_only, mean_err = _compare_eigen_sets(fin_drv_ml, fin_drv_ph, tol=1e-6)
        ml_unmatched, ph_unmatched = _compute_unmatched(fin_drv_ml, fin_drv_ph, tol=1e-6)

        print(
            "Sweep result: "
            f"Ks={gen0_ks_scale:.3f} Ka={gen0_ka_scale:.3f} tA={gen0_ta_scale:.3f} | "
            f"ML margin={margin_ml:+.6e} unstable={n_unstable_ml} stable={stable_ml} | "
            f"PH margin={margin_ph:+.6e} unstable={n_unstable_ph} stable={stable_ph}"
        )
        print(f"{grid_filename} small-signal (RmsProblemMultilinear)")
        print(f"  modes_driver_ml={len(fin_drv_ml)} stable_ml={stable_ml} margin_ml={margin_ml:.6e}")
        print(f"{grid_filename} small-signal (RmsProblemPhasor via driver)")
        print(f"  modes_driver_ph={len(fin_drv_ph)}")
        print("Driver eigen comparison (ML vs Phasor)")
        print(f"  matches={matches} ml_only={ml_only} ph_only={ph_only} mean_err={mean_err:.3e}")
        _print_participation_summary(problem_ml, eig_driver_ml, pf_driver_ml, label="ML driver")
        _print_participation_summary(problem_ph, eig_driver_ph, pf_driver_ph, label="Phasor driver")
        _print_unmatched(ml_unmatched, ph_unmatched, n_show=8)
        _plot_eigen_comparison(
            fin_drv_ml,
            fin_drv_ph,
            f"{grid_filename} Gen0 Ks x{gen0_ks_scale:.2f} Ka x{gen0_ka_scale:.2f} tA x{gen0_ta_scale:.2f}",
        )


if __name__ == "__main__":
    main()
