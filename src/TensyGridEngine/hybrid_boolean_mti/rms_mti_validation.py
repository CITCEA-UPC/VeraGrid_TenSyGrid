# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sys, os
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np

project_base = Path(__file__).resolve().parents[2]
src_path = project_base / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import VeraGridEngine.api as vge
from VeraGridEngine.Devices.Events.rms_event import RmsEvent
from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
from VeraGridEngine.Simulations.Rms.numerical.back_euler_mti import BackEulerImplicitIntegrationMTI
from VeraGridEngine.Simulations.Rms.numerical.back_euler_fx import BackEulerImplicitIntegration
from VeraGridEngine.Simulations.Rms.problems.rms_problem_MTI import RmsProblemMTI
from VeraGridEngine.Simulations.Rms.problems.rms_problem_phasor import RmsProblemPhasor
from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
from VeraGridEngine.Utils.Symbolic.bus_rms_template import initialize_bus_rms
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from VeraGridEngine.enumerations import (
    VarPowerFlowReferenceType,
    ParamPowerFlowReferenceType,
    RmsInitializationMethod,
)

os.environ["RMS_MTI_DEBUG"] = "1"

def build_grid_case() -> vge.MultiCircuit:
    grid_folder = project_base / "Grids_and_profiles" / "grids"
    grid_path = grid_folder / "IEEE 9 Bus.gridcal"
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    return vge.open_file(str(grid_path))


def build_rms_models_explicit(grid: vge.MultiCircuit) -> None:
    ensure_unique_device_names(grid)

    for bus in grid.buses:
        if bus.rms_model.empty():
            vge.initialize_bus_phasor_rms(bus, vf=grid.var_factory)

    for igen, gen in enumerate(grid.generators):
        if not gen.rms_model.empty():
            continue
        genqec = vge.get_complete_generator_template_phasor(
            grid.var_factory,
            name=f"Gen{igen}",
            hard_sat_type="mti",
        ).block
        genqec.connect([genqec.in_vars[0]], [gen.bus.rms_model.out_vars[0]])
        genqec.connect([genqec.in_vars[1]], [gen.bus.rms_model.out_vars[1]])
        genqec = vge.to_implicit(genqec, grid.var_factory)
        set_rms_model(device=gen, model=genqec, var_factory=grid.var_factory)

    for line in grid.lines:
        if not line.rms_model.empty():
            continue
        lineec = vge.get_line_phasor_rms_template(grid.var_factory, name=line.name).block
        lineec.connect([lineec.in_vars[0]], [line.bus_from.rms_model.out_vars[0]])
        lineec.connect([lineec.in_vars[1]], [line.bus_from.rms_model.out_vars[1]])
        lineec.connect([lineec.in_vars[2]], [line.bus_to.rms_model.out_vars[0]])
        lineec.connect([lineec.in_vars[3]], [line.bus_to.rms_model.out_vars[1]])
        lineec = vge.to_implicit(lineec, grid.var_factory)
        set_rms_model(device=line, model=lineec, var_factory=grid.var_factory)

    for load in grid.loads:
        if not load.rms_model.empty():
            continue
        loadec = vge.get_load_phasor_current_rms_template(grid.var_factory, name=load.name).block
        loadec.connect([loadec.in_vars[0]], [load.bus.rms_model.out_vars[0]])
        loadec.connect([loadec.in_vars[1]], [load.bus.rms_model.out_vars[1]])
        set_rms_model(device=load, model=loadec, var_factory=grid.var_factory)

    for trafo in grid.transformers2w:
        if not trafo.rms_model.empty():
            continue
        trafoec = vge.initialize_trafo_rms(trafo, grid.var_factory, use_phasor_template=True).block
        trafoec.connect([trafoec.in_vars[0]], [trafo.bus_from.rms_model.out_vars[0]])
        trafoec.connect([trafoec.in_vars[1]], [trafo.bus_from.rms_model.out_vars[1]])
        trafoec.connect([trafoec.in_vars[2]], [trafo.bus_to.rms_model.out_vars[0]])
        trafoec.connect([trafoec.in_vars[3]], [trafo.bus_to.rms_model.out_vars[1]])
        trafoec = vge.to_implicit(trafoec, grid.var_factory)
        set_rms_model(device=trafo, model=trafoec, var_factory=grid.var_factory)

    for shunt in grid.shunts:
        if not shunt.rms_model.empty():
            continue
        shuntec = vge.get_shunt_template(grid.var_factory, name=shunt.name, phasor=True).block
        shuntec.connect([shuntec.in_vars[0]], [shunt.bus.rms_model.out_vars[0]])
        shuntec.connect([shuntec.in_vars[1]], [shunt.bus.rms_model.out_vars[1]])
        shuntec = vge.to_implicit(shuntec, grid.var_factory)
        set_rms_model(device=shunt, model=shuntec, var_factory=grid.var_factory)

    sanitize_runtime_parameter_names(grid)


def sanitize_runtime_parameter_names(grid):
    def _sanitize_block(block, tag):
        block.unify_blocks()
        all_syms = list(block.algebraic_vars) + list(block.state_vars) + list(block.diff_vars)
        all_syms += list(block.event_dict.keys()) + list(block.mode_dict.keys())
        seen = {}
        for k, sym in enumerate(all_syms):
            nm = str(sym.name)
            if nm not in seen:
                seen[nm] = 0
                continue
            seen[nm] += 1
            sym.name = f"{nm}__{tag}_{seen[nm]}_{k}"

    for i, dev in enumerate(grid.get_branches_iter(add_vsc=True, add_hvdc=True, add_switch=True)):
        if not dev.rms_model.empty():
            _sanitize_block(dev.rms_model, f"br{i}")

    for i, dev in enumerate(grid.get_injection_devices_iter()):
        if not dev.rms_model.empty():
            _sanitize_block(dev.rms_model, f"inj{i}")


def ensure_unique_device_names(grid):
    def _rename(devices, prefix):
        seen = {}
        for i, dev in enumerate(devices):
            base = str(dev.name).strip() if getattr(dev, 'name', None) else f"{prefix}_{i}"
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


def inject_mti_boolean_logic(grid: vge.MultiCircuit, use_mode_dict_coupling: bool) -> tuple[object, object]:
    bus0 = grid.buses[0]
    mode_var = grid.var_factory.add_var(name="mti_mode_test_bus0")
    # For phasor-based MTI (RmsProblemMTI <- RmsProblemPhasor), runtime
    # parameters are registered from event_dict. Keep mode_dict optional so
    # we can test whether mode_dict coupling destabilizes the Jacobian.
    # Drive boolean logic from a continuous signal (bus voltage magnitude)
    # instead of self-referencing the mode variable.
    vr = bus0.rms_model.out_vars[0]
    vi = bus0.rms_model.out_vars[1]
    vm_sq = vr * vr + vi * vi
    vth = 0.98
    vth_sq = vth * vth

    # Guard residual is true when Vm <= Vth (low-voltage region).
    low_voltage_guard = vm_sq <= vth_sq
    bus0.rms_model.boolean_guards[mode_var] = low_voltage_guard
    bus0.rms_model.inequalities.append(low_voltage_guard)
    return bus0, mode_var


def add_proper_runtime_event(grid: vge.MultiCircuit) -> tuple[RmsEventsGroup, object, float, float]:

    loads = list(grid.loads)
    if len(loads) == 0:
        raise RuntimeError("No loads available for RMS event injection.")
    target = loads[0]
    if len(target.rms_model.event_dict) == 0:
        raise RuntimeError("Selected load model has no runtime event parameters.")

    event_param = target.rms_model.api_obj_mapping.get(ParamPowerFlowReferenceType.Pl0, None)
    if event_param is None or event_param not in target.rms_model.event_dict:
        raise RuntimeError("Could not find load Pl0 runtime parameter in event_dict.")
    base_expr = target.rms_model.event_dict[event_param]
    event_value = 0.98

    base_expr_value = None
    if hasattr(base_expr, "value") and base_expr.value is not None:
        try:
            base_expr_value = float(base_expr.value)
        except Exception:
            base_expr_value = None

    base_load_pu = None
    if hasattr(target, "P"):
        try:
            sbase = float(getattr(grid, "Sbase", 100.0))
            if abs(sbase) > 1e-12:
                base_load_pu = float(target.P) / sbase
        except Exception:
            base_load_pu = None

    if base_expr_value is not None and np.isfinite(base_expr_value) and abs(base_expr_value) > 1e-12:
        event_value = 0.99 * base_expr_value
    elif base_load_pu is not None and np.isfinite(base_load_pu):
        event_value = 0.95 * base_load_pu

    event_group = RmsEventsGroup(name="MTIValidationEventGroup")
    grid.add_rms_events_group(event_group)
    event_time = 0.05
    grid.add_rms_event(
        RmsEvent(
            device=target,
            parameter=event_param,
            time=event_time,
            value=-float(event_value),
            group=event_group,
            force_step_alignment=True,
        )
    )
    return event_group, target, event_time, float(event_value)


def ensure_problem_event_api(problem) -> None:
    if not hasattr(problem, "get_next_forced_event_time"):
        def _get_next_forced_event_time(self, t_local_prev, t_macro_target):
            return None
        problem.get_next_forced_event_time = MethodType(_get_next_forced_event_time, problem)

    if not hasattr(problem, "update"):
        def _update(self, t, x_snapshot, variable_parameters):
            return None
        problem.update = MethodType(_update, problem)


def get_hard_sat_event_limits(grid: vge.MultiCircuit, sat_var_name: str) -> tuple[float, float] | None:
    for dev in list(grid.get_injection_devices_iter()) + list(grid.get_branches_iter(add_vsc=True, add_hvdc=True, add_switch=True)):
        block = getattr(dev, "rms_model", None)
        if block is None or block.empty():
            continue

        block.unify_blocks()
        if not any(str(var.name) == sat_var_name for var in block.algebraic_vars):
            continue

        uc_var = next((var for var in block.event_dict if str(var.name) == "Uc"), None)
        uo_var = next((var for var in block.event_dict if str(var.name) == "Uo"), None)
        if uc_var is None or uo_var is None:
            continue

        uc = block.event_dict[uc_var]
        uo = block.event_dict[uo_var]
        if uc.value is None or uo.value is None:
            continue

        return float(uc.value), float(uo.value)

    return None


def get_problem_event_limits(problem: RmsProblemPhasor, lower_name: str, upper_name: str) -> tuple[float, float] | None:
    params = getattr(problem, "_variable_parameters", [])
    eqs = getattr(problem, "_event_parameters_eqs", [])
    lower = None
    upper = None
    for i, var in enumerate(params):
        if i >= len(eqs):
            continue
        expr = eqs[i]
        value = getattr(expr, "value", None)
        if value is None:
            continue
        name = str(getattr(var, "name", ""))
        if name == lower_name and lower is None:
            lower = float(value)
        elif name == upper_name and upper is None:
            upper = float(value)
        if lower is not None and upper is not None:
            return lower, upper
    return None


def plot_results(
    t: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    problem: RmsProblemPhasor,
    grid: vge.MultiCircuit,
    event_load: object | None,
    event_time: float,
    event_value: float,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)

    vr_entries = []
    vi_entries = []
    for uid, idx in problem.uid2idx_vars.items():
        var = problem.sys_vars[uid]
        if not hasattr(var, "ref"):
            continue
        if var.ref == VarPowerFlowReferenceType.Vr:
            vr_entries.append((idx, str(var.name)))
        elif var.ref == VarPowerFlowReferenceType.Vi:
            vi_entries.append((idx, str(var.name)))

    vr_entries.sort(key=lambda x: x[0])
    vi_entries.sort(key=lambda x: x[0])

    if vr_entries:
        for idx, name in vr_entries:
            axes[0].plot(t, y[:, idx], linewidth=1.7, label=name)
        axes[0].set_ylabel("Vr [pu]")
        axes[0].legend(loc="best", ncol=2, fontsize=8)
    else:
        axes[0].plot(t, np.zeros_like(t), linewidth=1.5)
        axes[0].set_ylabel("Vr [pu]")
    axes[0].grid(True, alpha=0.3)

    if vi_entries:
        for idx, name in vi_entries:
            axes[1].plot(t, y[:, idx], linewidth=1.7, label=name)
        axes[1].set_ylabel("Vi [pu]")
        axes[1].legend(loc="best", ncol=2, fontsize=8)
    else:
        axes[1].plot(t, np.zeros_like(t), linewidth=1.5)
        axes[1].set_ylabel("Vi [pu]")
    axes[1].grid(True, alpha=0.3)

    if z.shape[1] > 0:
        for k in range(z.shape[1]):
            axes[2].step(t, z[:, k], where="post", linewidth=2, label=f"z{k}")
        axes[2].legend(loc="best")
    else:
        axes[2].plot(t, np.zeros_like(t), linewidth=1.5)
    axes[2].set_ylabel("Boolean z")
    axes[2].grid(True, alpha=0.3)

    loads = list(grid.loads)
    if len(loads) > 0:
        for k, load in enumerate(loads):
            p0 = float(getattr(load, "P", 0.0))
            p_hist = np.full_like(t, p0, dtype=float)
            if event_load is load:
                p_hist[t >= event_time] = event_value
            label = getattr(load, "name", f"load{k}")
            axes[3].step(t, p_hist, where="post", linewidth=1.8, label=f"{label}")
        axes[3].legend(loc="best", ncol=2, fontsize=8)
    else:
        axes[3].plot(t, np.zeros_like(t), linewidth=1.5)
    axes[3].set_ylabel("Pl [pu]")
    axes[3].grid(True, alpha=0.3)

    name_to_idx = {}
    named_entries = []
    for uid, idx in problem.uid2idx_vars.items():
        name = str(problem.sys_vars[uid].name)
        name_to_idx[name] = idx
        named_entries.append((idx, name))

    sat_targets = []
    for preferred in ("y_sat_gov_rate", "hs_gov_rate", "y_gov1_rate", "u_gov1"):
        matches = sorted((idx, name) for idx, name in named_entries if name == preferred)
        if matches:
            sat_targets = matches
            break
    if not sat_targets:
        sat_targets = sorted((idx, name) for idx, name in named_entries if name.startswith("y_sat_"))[:3]

    if sat_targets:
        limits = get_problem_event_limits(problem, "Uc", "Uo")
        if limits is None:
            limits = get_hard_sat_event_limits(grid, sat_targets[0][1])
        for idx_val, key in sat_targets:
            y_val = y[:, idx_val]
            axes[4].plot(t, y_val, linewidth=2.0, label=f"{key}[{idx_val}]", zorder=3)
        all_vals = np.concatenate([np.asarray(y[:, idx], dtype=float) for idx, _ in sat_targets])
        if limits is not None:
            y_min_lim, y_max_lim = limits
            axes[4].axhline(float(y_max_lim), color="tab:red", linestyle="--", linewidth=1.6, label=f"Uo={y_max_lim:g}")
            axes[4].axhline(float(y_min_lim), color="tab:green", linestyle="--", linewidth=1.6, label=f"Uc={y_min_lim:g}")
            pad = max(0.02 * max(abs(float(y_min_lim)), abs(float(y_max_lim)), 1.0), 1e-4)
            axes[4].set_ylim(float(y_min_lim) - pad, float(y_max_lim) + pad)
        else:
            pad = max(0.02 * max(float(np.max(np.abs(all_vals))), 1.0), 1e-4)
            axes[4].set_ylim(float(np.min(all_vals)) - pad, float(np.max(all_vals)) + pad)
        axes[4].legend(loc="best", ncol=2, fontsize=8)
    else:
        axes[4].plot(t, np.zeros_like(t), linewidth=1.5)
        axes[4].text(0.01, 0.85, "No governor saturation variable found", transform=axes[4].transAxes)
    axes[4].set_ylabel("Saturation")
    axes[4].set_xlabel("Time [s]")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    out = project_base / "trunk" / "tensygrid" / "rms_mti_validation_plot.png"
    plt.savefig(out, dpi=150)
    print(f"Saved plot: {out}")


def run_validation() -> None:
    inject_boolean_logic = False
    use_mode_dict_coupling = False
    grid = build_grid_case()
    # Use the same explicit phasor-style model build strategy as
    # small_signal_tensygrid.py
    build_rms_models_explicit(grid)
    if inject_boolean_logic:
        inject_mti_boolean_logic(grid, use_mode_dict_coupling=use_mode_dict_coupling)
    events_group, event_load, event_time, event_value = add_proper_runtime_event(grid)

    pf_options = vge.PowerFlowOptions(tolerance=1e-6)
    pf_results = vge.power_flow(grid, pf_options)
    if not pf_results.converged:
        raise RuntimeError("Power flow did not converge; aborting MTI validation.")

    rms_options = RmsOptions(
        time_step=0.01,
        simulation_time=2.00,
        tolerance=1e-6,
        max_iter=20,
        verbose=1,
        initialization_method=RmsInitializationMethod.Explicit,
    )

    #problem_phasor = RmsProblemPhasor(grid=grid, options=rms_options, pf_results=pf_results)
    problem = RmsProblemMTI(grid=grid, options=rms_options, pf_results=pf_results)
    problem.set_events_group(events_group)
    ensure_problem_event_api(problem)

    def report_init_guess_vs_x0(problem_obj: RmsProblemPhasor, atol: float = 1e-10) -> None:
        x0_local = np.asarray(problem_obj.get_x0(), dtype=float)
        pairs = []
        missing = 0
        for uid, guess_val in problem_obj.init_guess.items():
            idx = problem_obj.uid2idx_vars.get(uid, None)
            if idx is None:
                missing += 1
                continue
            x0_val = float(x0_local[idx])
            gval = float(guess_val)
            diff = abs(x0_val - gval)
            var_name = str(problem_obj.sys_vars[uid].name) if uid in problem_obj.sys_vars else str(uid)
            pairs.append((diff, idx, uid, var_name, gval, x0_val))

        if len(pairs) == 0:
            print("\n[INIT-CHECK] No init_guess entries mapped to x0 variables.")
            return

        diffs = np.asarray([p[0] for p in pairs], dtype=float)
        n_total = len(pairs)
        n_match = int(np.sum(diffs <= atol))
        n_mismatch = n_total - n_match
        print("\n[INIT-CHECK] init_guess vs x0")
        print(f"  compared={n_total}, matched(|diff|<={atol:.1e})={n_match}, mismatched={n_mismatch}, missing_uid_map={missing}")
        print(f"  max|diff|={float(np.max(diffs)):.6e}, mean|diff|={float(np.mean(diffs)):.6e}")

        if n_mismatch > 0:
            pairs.sort(key=lambda x: x[0], reverse=True)
            top = pairs[:20]
            print("  top mismatches:")
            for diff, idx, uid, var_name, gval, x0_val in top:
                if diff <= atol:
                    break
                print(
                    f"    idx={idx:4d} uid={uid} var={var_name} "
                    f"init_guess={gval:+.6e} x0={x0_val:+.6e} diff={diff:.6e}"
                )

    report_init_guess_vs_x0(problem)

    def report_phasor_vs_mti_x0(
        phasor_obj: RmsProblemPhasor,
        mti_obj: RmsProblemPhasor,
        atol: float = 1e-10,
    ) -> None:
        x0_ph = np.asarray(phasor_obj.get_x0(), dtype=float)
        x0_mti = np.asarray(mti_obj.get_x0(), dtype=float)

        common_uids = sorted(set(phasor_obj.uid2idx_vars.keys()) & set(mti_obj.uid2idx_vars.keys()))
        if len(common_uids) == 0:
            print("\n[X0-COMPARE] No common variable UIDs between Phasor and MTI.")
            return

        rows = []
        for uid in common_uids:
            idx_ph = phasor_obj.uid2idx_vars[uid]
            idx_mti = mti_obj.uid2idx_vars[uid]
            vph = float(x0_ph[idx_ph])
            vmti = float(x0_mti[idx_mti])
            diff = abs(vph - vmti)
            name = str(mti_obj.sys_vars[uid].name) if uid in mti_obj.sys_vars else str(uid)
            rows.append((diff, uid, idx_ph, idx_mti, name, vph, vmti))

        diffs = np.asarray([r[0] for r in rows], dtype=float)
        n_total = len(rows)
        n_match = int(np.sum(diffs <= atol))
        n_mismatch = n_total - n_match

        print("\n[X0-COMPARE] RmsProblemPhasor vs RmsProblemMTI")
        print(f"  compared={n_total}, matched(|diff|<={atol:.1e})={n_match}, mismatched={n_mismatch}")
        print(f"  max|diff|={float(np.max(diffs)):.6e}, mean|diff|={float(np.mean(diffs)):.6e}")

        if n_mismatch > 0:
            rows.sort(key=lambda x: x[0], reverse=True)
            print("  top mismatches:")
            for diff, uid, idx_ph, idx_mti, name, vph, vmti in rows[:20]:
                if diff <= atol:
                    break
                print(
                    f"    uid={uid} var={name} "
                    f"phasor[idx={idx_ph}]={vph:+.6e} mti[idx={idx_mti}]={vmti:+.6e} diff={diff:.6e}"
                )

    #report_phasor_vs_mti_x0(problem_phasor, problem)

    solver = BackEulerImplicitIntegrationMTI(
        problem=problem,
        t0=0.0,
        t_end=rms_options.simulation_time,
        h=rms_options.time_step,
        max_iter=rms_options.max_iter,
        tolerance=rms_options.tolerance,
        inequality_tolerance=1e-9,
    )

    t, y, well_initialized, converged = solver.simulate()

    print("\n=== MTI Validation Report ===")
    print(f"Boolean injection enabled: {inject_boolean_logic}")
    print(f"Mode-dict coupling enabled: {use_mode_dict_coupling}")
    print(f"Power flow converged: {pf_results.converged}")
    print(f"RMS variables: {problem.get_all_vars_number()}")
    print(f"Boolean modes registered: {len(problem.get_mti_boolean_parameter_indices)}")
    print(f"Well initialized: {well_initialized}")
    print(f"Converged: {converged}")
    print(f"Simulated steps: {len(t)}")
    print(f"Final state norm: {np.linalg.norm(y[-1, :]):.6e}")

    vr_entries = []
    vi_entries = []
    for uid, idx in problem.uid2idx_vars.items():
        var = problem.sys_vars[uid]
        if not hasattr(var, "ref"):
            continue
        if var.ref == VarPowerFlowReferenceType.Vr:
            vr_entries.append((idx, str(var.name)))
        elif var.ref == VarPowerFlowReferenceType.Vi:
            vi_entries.append((idx, str(var.name)))
    vr_entries.sort(key=lambda x: x[0])
    vi_entries.sort(key=lambda x: x[0])

    if vr_entries or vi_entries:
        print("\nFinal phasor voltages (Vr, Vi):")
        for i in range(max(len(vr_entries), len(vi_entries))):
            vr_txt = "-"
            vi_txt = "-"
            if i < len(vr_entries):
                vr_idx, vr_name = vr_entries[i]
                vr_txt = f"{vr_name}={y[-1, vr_idx]:.6f}"
            if i < len(vi_entries):
                vi_idx, vi_name = vi_entries[i]
                vi_txt = f"{vi_name}={y[-1, vi_idx]:.6f}"
            print(f"  {vr_txt}, {vi_txt}")

    z_hist = solver.z[: len(t), :] if solver.z.shape[1] > 0 else solver.z
    if z_hist.shape[1] > 0:
        print(f"Final z: {z_hist[-1, :]}")
        print(f"Unique z values: {np.unique(z_hist, axis=0)}")

    plot_results(t, y, z_hist, problem, grid, event_load, event_time, event_value)


if __name__ == "__main__":
    run_validation()
