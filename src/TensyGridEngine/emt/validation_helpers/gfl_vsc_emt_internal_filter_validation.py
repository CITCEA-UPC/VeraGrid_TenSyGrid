# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.vsc_gfl_emt import VscGflEmtBuild
from VeraGridEngine.Utils.Symbolic.block import Block, Const, VarPowerFlowReferenceType
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Templates.Emt.dc_source_emt_template import get_dc_voltage_source_emt_template
from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import get_generator_thevenin_rl_emt_template_with_ref
from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.initialization_emt import run_emt_native_initialization
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
import VeraGridEngine.Utils.Symbolic.symbolic as sym
from VeraGridEngine.Utils.Symbolic.diagnostic import NewtonDiagnosticsConfig
from VeraGridEngine.enumerations import ConverterControlType, DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes
import VeraGridEngine.api as gce


PLOT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "gfl_vsc_emt_internal_filter_validation_plots")
_DEFAULT_INTEGRATION_METHOD_NAME = os.environ.get("VERAGRID_GFL_EMT_INTEGRATION_METHOD", "DaeTrapezoidal")
_DEFAULT_KP_ICL = "0.1" if _DEFAULT_INTEGRATION_METHOD_NAME == "DaeTrapezoidal" else "1.0"
_DEFAULT_KI_ICL = "2.0" if _DEFAULT_INTEGRATION_METHOD_NAME == "DaeTrapezoidal" else "20.0"
Q_CONTROL_KP_POL = float(os.environ.get("VERAGRID_GFL_EMT_KP_POL", "0.6"))
Q_CONTROL_KI_POL = float(os.environ.get("VERAGRID_GFL_EMT_KI_POL", "10.0"))
CURRENT_CONTROL_KP_ICL = float(os.environ.get("VERAGRID_GFL_EMT_KP_ICL", _DEFAULT_KP_ICL))
CURRENT_CONTROL_KI_ICL = float(os.environ.get("VERAGRID_GFL_EMT_KI_ICL", _DEFAULT_KI_ICL))
PLL_KP = float(os.environ.get("VERAGRID_GFL_EMT_KP_PLL", "0.01"))
PLL_KI = float(os.environ.get("VERAGRID_GFL_EMT_KI_PLL", "0.05"))
DC_LINK_CAPACITANCE = float(os.environ.get("VERAGRID_GFL_EMT_CDC", "10.0"))
VERAGRID_GFL_EMT_TIME_STEP = float(os.environ.get("VERAGRID_GFL_EMT_TIME_STEP", "1e-4"))
VERAGRID_GFL_EMT_SIM_TIME = float(os.environ.get("VERAGRID_GFL_EMT_SIM_TIME", "1.0"))
VERAGRID_GFL_EMT_WARMUP_TIME = float(os.environ.get("VERAGRID_GFL_EMT_WARMUP_TIME", "0.0"))
P_REF_EVENT_ENABLED = os.environ.get("VERAGRID_GFL_EMT_PREF_EVENT", "1") != "0"
P_REF_EVENT_TIME = float(os.environ.get("VERAGRID_GFL_EMT_PREF_EVENT_TIME", "0.25"))
P_REF_EVENT_DELTA = float(os.environ.get("VERAGRID_GFL_EMT_PREF_EVENT_DELTA", "0.000"))
INIT_DIAGNOSTIC_TOP_N = int(os.environ.get("VERAGRID_GFL_EMT_INIT_DIAG_TOP_N", "25"))
POST_SEED_INITIALIZATION_METHOD_NAME = os.environ.get("VERAGRID_GFL_EMT_POST_SEED_INIT", "")


def find_name_in_block(name: str, block: Block):
    """Recursively find a variable/event parameter by exact name, then by generated prefix."""

    def find_match(search_block: Block, allow_prefix: bool):
        block_vars = (
            search_block.algebraic_vars
            + search_block.state_vars
            + search_block.diff_vars
            + search_block.in_vars
            + search_block.out_vars
            + list(search_block.event_dict.keys())
            + list(search_block.parameters.keys())
        )
        for var in block_vars:
            var_name = getattr(var, 'name', None)
            if (
                var_name is not None
                and hasattr(var, 'non_mutable_uid')
                and (name == var_name or (allow_prefix and var_name.startswith(f"{name}_")))
            ):
                return var

        for mdl in search_block.children:
            res = find_match(mdl, allow_prefix)
            if res is not None:
                return res

        return None

    exact = find_match(block, allow_prefix=False)
    if exact is not None:
        return exact

    return find_match(block, allow_prefix=True)


def set_event_value_in_block(vf, name: str, block: Block, value: float) -> bool:
    for var in block.event_dict.keys():
        if name == var.name:
            block.event_dict[var] = vf.add_const(float(value))
            return True

    for mdl in block.children:
        if set_event_value_in_block(vf, name, mdl, value):
            return True

    return False


def set_internal_runtime_value(problem, block: Block, name: str, value: float) -> None:
    """Override a named runtime parameter for this validation experiment.

    Update the owning model as well as the solver's runtime stores so a later
    parameter refresh retains the override. Missing parameters are ignored.
    """
    for owner in block.get_all_blocks():
        for parameter in owner.event_dict:
            if parameter.name == name:
                owner.event_dict[parameter] = Const(float(value))
                source_parameters = problem._runtime_all_parameters_source
                source_equations = problem._runtime_all_eqs_source
                if isinstance(source_parameters, list) and isinstance(source_equations, list):
                    for index, source_parameter in enumerate(source_parameters):
                        if source_parameter.uid == parameter.uid:
                            source_equations[index] = Const(float(value))
                problem._set_runtime_parameter_value_by_uid(parameter.uid, float(value))
                return


def set_problem_runtime_value(problem: EmtProblemDae, vf, block: Block, name: str, value: float) -> None:
    set_internal_runtime_value(problem, block, name, value)
    var = find_name_in_block(name, block)
    uid2idx = getattr(problem, "_uid2idx_event_params", getattr(problem, "uid2idx_event_params", {}))
    runtime_vars = []
    if var is not None and var.uid in uid2idx:
        runtime_vars.append(var)

    for candidate in problem.get_variable_parameters():
        candidate_name = getattr(candidate, "name", "")
        if candidate_name == name or candidate_name.startswith(f"{name}_"):
            if candidate.uid in uid2idx and all(existing.uid != candidate.uid for existing in runtime_vars):
                runtime_vars.append(candidate)

    if not runtime_vars:
        return

    const_value = vf.add_const(float(value))
    for runtime_var in runtime_vars:
        idx = uid2idx[runtime_var.uid]
        problem._event_params_values[idx] = float(value)
        problem._event_parameters_eqs[idx] = const_value
        if hasattr(problem, "_runtime_parameter_eqs0"):
            problem._runtime_parameter_eqs0[idx] = const_value
        if hasattr(problem, "_runtime_all_eqs_source"):
            problem._runtime_all_eqs_source[idx] = const_value
        if hasattr(problem, "_runtime_continuous_slice") and hasattr(problem, "_runtime_continuous_eqs"):
            continuous_slice = problem._runtime_continuous_slice
            if continuous_slice.start <= idx < continuous_slice.stop:
                problem._runtime_continuous_eqs[idx - continuous_slice.start] = const_value
        if hasattr(problem, "_runtime_mode_slice") and hasattr(problem, "_runtime_mode_eqs"):
            mode_slice = problem._runtime_mode_slice
            if mode_slice.start <= idx < mode_slice.stop:
                problem._runtime_mode_eqs[idx - mode_slice.start] = const_value


def pf_voltage_abc(grid: gce.MultiCircuit, pf_results, bus: gce.Bus) -> tuple[float, float, float]:
    v_complex = complex(pf_results.voltage[grid.buses.index(bus)])
    v_peak = np.sqrt(2.0) * abs(v_complex)
    v_angle = np.angle(v_complex)
    return (
        v_peak * np.sin(v_angle),
        v_peak * np.sin(v_angle - 2.0 * np.pi / 3.0),
        v_peak * np.sin(v_angle + 2.0 * np.pi / 3.0),
    )


def park_dq_value(theta_value: float, va_value: float, vb_value: float, vc_value: float) -> tuple[float, float]:
    cos_t = np.cos(theta_value)
    sin_t = np.sin(theta_value)
    sqrt3 = np.sqrt(3.0)
    vd_value = (1.0 / 3.0) * (
        2.0 * cos_t * va_value
        + (-cos_t - sqrt3 * sin_t) * vb_value
        + (-cos_t + sqrt3 * sin_t) * vc_value
    )
    vq_value = (1.0 / 3.0) * (
        2.0 * sin_t * va_value
        + (-sin_t + sqrt3 * cos_t) * vb_value
        + (-sin_t - sqrt3 * cos_t) * vc_value
    )
    return vd_value, vq_value


def inverse_park_value(theta_value: float, vd_value: float, vq_value: float) -> tuple[float, float, float]:
    cos_t = np.cos(theta_value)
    sin_t = np.sin(theta_value)
    sqrt3 = np.sqrt(3.0)
    va_value = vd_value * cos_t + vq_value * sin_t
    vb_value = (
        vd_value * (-0.5 * cos_t - 0.5 * sqrt3 * sin_t)
        + vq_value * (-0.5 * sin_t + 0.5 * sqrt3 * cos_t)
    )
    vc_value = (
        vd_value * (-0.5 * cos_t + 0.5 * sqrt3 * sin_t)
        + vq_value * (-0.5 * sin_t - 0.5 * sqrt3 * cos_t)
    )
    return va_value, vb_value, vc_value


def set_init_if_exists(problem: EmtProblemDae, block: Block, name: str, value: float) -> None:
    var = find_name_in_block(name, block)
    if var is not None:
        problem.init_guess[var.uid] = float(value)


def trace_for_var(problem: EmtProblemDae, y: np.ndarray, var):
    if var is None or var.uid not in problem._uid2idx_vars:
        return None

    return y[:, problem.get_var_idx(var)]


def required_trace(problem: EmtProblemDae, y: np.ndarray, var, label: str) -> np.ndarray:
    trace = trace_for_var(problem, y, var)
    if trace is None:
        raise RuntimeError(f"Cannot plot {label}: variable is not in the solved EMT vector")
    return np.asarray(trace, dtype=float)


def finalize_time_axis(t: np.ndarray) -> None:
    plt.xlim(float(t[0]), float(t[-1]))


def save_plot(filename: str) -> None:
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, filename), dpi=150)
    plt.close()


def plot_traces(
    t: np.ndarray,
    problem: EmtProblemDae,
    y: np.ndarray,
    vsc_mdl: Block,
    bus_dc: gce.Bus,
    bus2: gce.Bus,
    p_ref0: float,
) -> None:
    t = np.asarray(t, dtype=float)
    v_a_b2 = required_trace(problem, y, find_name_in_block('v_A', bus2.emt_model), 'B2 v_A')
    v_b_b2 = required_trace(problem, y, find_name_in_block('v_B', bus2.emt_model), 'B2 v_B')
    v_c_b2 = required_trace(problem, y, find_name_in_block('v_C', bus2.emt_model), 'B2 v_C')
    plt.figure(figsize=(10, 6))
    plt.plot(t, v_a_b2, label="v_A")
    plt.plot(t, v_b_b2, label="v_B")
    plt.plot(t, v_c_b2, label="v_C")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (pu peak)")
    plt.title("B2 Phase Voltages")
    finalize_time_axis(t)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    save_plot("b2_phase_voltages.png")

    i_a = required_trace(problem, y, find_name_in_block('i_filter_A', vsc_mdl), 'i_filter_A')
    i_b = required_trace(problem, y, find_name_in_block('i_filter_B', vsc_mdl), 'i_filter_B')
    i_c = required_trace(problem, y, find_name_in_block('i_filter_C', vsc_mdl), 'i_filter_C')
    plt.figure(figsize=(10, 6))
    plt.plot(t, i_a, label="i_A")
    plt.plot(t, i_b, label="i_B")
    plt.plot(t, i_c, label="i_C")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (pu peak)")
    plt.title("VSC Internal Filter Currents")
    finalize_time_axis(t)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    save_plot("vsc_internal_filter_currents.png")

    p_trace = required_trace(problem, y, find_name_in_block('P', vsc_mdl), 'P')
    q_trace = required_trace(problem, y, find_name_in_block('Q', vsc_mdl), 'Q')
    plt.figure(figsize=(10, 6))
    plt.plot(t, p_trace, label="P")
    plt.plot(t, q_trace, label="Q")
    if P_REF_EVENT_ENABLED:
        p_ref_trace = np.where(t >= P_REF_EVENT_TIME, p_ref0 + P_REF_EVENT_DELTA, p_ref0)
        plt.step(t, p_ref_trace, where="post", label="P_ref", linestyle="--", color="tab:red")
    else:
        plt.axhline(p_ref0, label="P_ref", linestyle="--", color="tab:red")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (pu)")
    plt.title("VSC PCC Power")
    finalize_time_axis(t)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    save_plot("vsc_pq.png")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, required_trace(problem, y, find_name_in_block('vg_d', vsc_mdl), 'vg_d'), label="vg_d")
    axes[0].plot(t, required_trace(problem, y, find_name_in_block('vg_q', vsc_mdl), 'vg_q'), label="vg_q")
    axes[0].plot(t, required_trace(problem, y, find_name_in_block('v_d_c', vsc_mdl), 'v_d_c'), label="v_d_c", linestyle="--")
    axes[0].plot(t, required_trace(problem, y, find_name_in_block('v_q_c', vsc_mdl), 'v_q_c'), label="v_q_c", linestyle="--")
    axes[0].set_ylabel("Voltage (pu peak)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    axes[1].plot(t, required_trace(problem, y, find_name_in_block('i_line_d', vsc_mdl), 'i_line_d'), label="i_d")
    axes[1].plot(t, required_trace(problem, y, find_name_in_block('i_line_q', vsc_mdl), 'i_line_q'), label="i_q")
    axes[1].plot(t, required_trace(problem, y, find_name_in_block('i_d_ref', vsc_mdl), 'i_d_ref'), label="i_d_ref", linestyle="--")
    axes[1].plot(t, required_trace(problem, y, find_name_in_block('i_q_ref', vsc_mdl), 'i_q_ref'), label="i_q_ref", linestyle="--")
    axes[1].set_ylabel("Current (pu peak)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    vdc_trace = required_trace(problem, y, bus_dc.emt_model.out_vars[0], 'Vdc')
    axes[2].plot(t, vdc_trace, label="Vdc")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Voltage (pu)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True)
    axes[2].set_xlim(float(t[0]), float(t[-1]))
    fig.suptitle("VSC Internal Filter dq Variables and DC Voltage")
    fig.tight_layout()
    save_plot("vsc_dq_and_dc_voltage.png")


def print_problem_structure(problem: EmtProblemDae) -> None:
    state_vars = problem.get_state_vars()
    algebraic_vars = problem.get_algebraic_vars()
    state_eqs = problem.get_state_eqs()
    algebraic_eqs = problem.get_algebraic_eqs()
    state_uids = {var.uid for var in state_vars}
    alg_uids = {var.uid for var in algebraic_vars}
    duplicated = [var.name for var in algebraic_vars if var.uid in state_uids]
    print(
        "EMT structure: "
        f"states={len(state_vars)}, algebraic={len(algebraic_vars)}, "
        f"state_eqs={len(state_eqs)}, algebraic_eqs={len(algebraic_eqs)}, "
        f"unknowns={len(state_vars) + len(algebraic_vars)}, equations={len(state_eqs) + len(algebraic_eqs)}"
    )
    if duplicated:
        print(f"Variables present as both state and algebraic: {duplicated}")


def print_vsc_seed_values(problem: EmtProblemDae, vsc_mdl: Block) -> None:
    x0 = problem.get_x0()
    print("Post-seed VSC/filter x0 values:")
    for name in ('vg_d', 'vg_q', 'v_d_c', 'v_q_c', 'vc_d', 'vc_q', 'va_v', 'vb_v', 'vc_v', 'i_a_f', 'i_b_f', 'i_c_f'):
        var = find_name_in_block(name, vsc_mdl)
        idx = None if var is None or var.uid not in problem._uid2idx_vars else problem.get_var_idx(var)
        if idx is not None:
            print(f"  {name}={x0[idx]: .9e}")


def build_eval_uid_bindings(problem: EmtProblemDae, x: np.ndarray, dx: np.ndarray | None, t_value: float) -> dict[int, float]:
    bindings = {problem.glob_time.uid: float(t_value)}
    for var in problem.get_state_vars() + problem.get_algebraic_vars():
        idx = problem._uid2idx_vars.get(var.uid, None)
        if idx is not None:
            bindings[var.uid] = float(x[idx])

    if dx is not None:
        uid2idx_diff = getattr(problem, "uid2idx_diff", getattr(problem, "_uid2idx_diff", {}))
        for var in problem.get_diff_vars():
            idx = uid2idx_diff.get(var.uid, None)
            if idx is not None and idx < len(dx):
                bindings[var.uid] = float(dx[idx])

    runtime_params = problem.def_event_params_fn(problem.event_params_values.copy(), float(t_value))
    for var in problem.get_variable_parameters():
        idx = problem._uid2idx_event_params.get(var.uid, None)
        if idx is not None:
            bindings[var.uid] = float(runtime_params[idx])

    for var, value in zip(problem.get_constant_parameters(), problem.get_parameters_values()):
        bindings[var.uid] = float(value.value)

    return bindings


def eval_expr_with_bindings(expr, bindings: dict[int, float]) -> float:
    if hasattr(expr, "eval_uid"):
        return float(expr.eval_uid(bindings))
    return float(expr)


def evaluate_state_rhs(problem: EmtProblemDae, x: np.ndarray, dx: np.ndarray | None, t_value: float) -> np.ndarray:
    bindings = build_eval_uid_bindings(problem, x, dx, t_value)
    rhs = np.zeros(len(problem.get_state_eqs()), dtype=float)
    for i, eq in enumerate(problem.get_state_eqs()):
        rhs[i] = eval_expr_with_bindings(eq, bindings)
    return rhs


def evaluate_algebraic_residuals(problem: EmtProblemDae, x: np.ndarray, dx: np.ndarray | None, t_value: float) -> np.ndarray:
    bindings = build_eval_uid_bindings(problem, x, dx, t_value)
    residuals = np.zeros(len(problem.get_algebraic_eqs()), dtype=float)
    for i, eq in enumerate(problem.get_algebraic_eqs()):
        residuals[i] = eval_expr_with_bindings(eq, bindings)
    return residuals


def print_ranked_diagnostics(title: str, rows: list[tuple[float, str]], top_n: int = INIT_DIAGNOSTIC_TOP_N) -> None:
    print(title)
    if not rows:
        print("  <none>")
        return

    for _, text in sorted(rows, key=lambda item: item[0], reverse=True)[:top_n]:
        print(f"  {text}")


def wrapped_angle_delta(value_end: float, value_start: float) -> float:
    return float((value_end - value_start + np.pi) % (2.0 * np.pi) - np.pi)


def print_initialization_diagnostics(problem: EmtProblemDae, vsc_mdl: Block, t: np.ndarray, y: np.ndarray, dy: np.ndarray, x0_seed: np.ndarray, dx0_seed: np.ndarray) -> None:
    if y.size == 0:
        return

    x_accepted0 = np.asarray(y[0, :], dtype=float)
    x_first = np.asarray(y[min(1, y.shape[0] - 1), :], dtype=float)
    x_last = np.asarray(y[-1, :], dtype=float)
    dx_accepted0 = np.asarray(dy[0, :], dtype=float) if dy is not None and dy.size else dx0_seed
    dx_first = np.asarray(dy[min(1, dy.shape[0] - 1), :], dtype=float) if dy is not None and dy.size else dx0_seed
    dx_last = np.asarray(dy[-1, :], dtype=float) if dy is not None and dy.size else dx0_seed
    t0 = float(t[0])
    t_first = float(t[min(1, len(t) - 1)])
    t_last = float(t[-1])

    print("Initialization diagnostics:")
    print(f"  ||accepted x0 - seed x0||_inf = {np.max(np.abs(x_accepted0 - x0_seed)):.9e}")
    print(f"  ||xlast - seed x0||_inf      = {np.max(np.abs(x_last - x0_seed)):.9e}")
    print(f"  ||accepted dx0 - seed dx0||_inf = {np.max(np.abs(dx_accepted0 - dx0_seed)):.9e}")

    focused_names = (
        "i_line_q", "i_q_ref", "u_Pac_ctrl", "xi_Pac_ctrl", "P", "P_ref",
        "i_line_d", "i_d_ref", "u_Qac_ctrl", "xi_Qac_ctrl", "Q", "Q_ref",
    )
    focused_rows = []
    focused_indices = []
    for name in focused_names:
        var = find_name_in_block(name, vsc_mdl)
        idx = None if var is None else problem._uid2idx_vars.get(var.uid, None)
        if idx is not None:
            focused_indices.append((name, idx))
            focused_rows.append(
                f"{name}: x0={x0_seed[idx]: .9e}, x1={x_first[idx]: .9e}, "
                f"xlast={x_last[idx]: .9e}, x1-x0={x_first[idx] - x0_seed[idx]: .9e}, "
                f"xlast-x0={x_last[idx] - x0_seed[idx]: .9e}"
            )
    print("Focused VSC control start/end values:")
    for row in focused_rows:
        print(f"  {row}")

    if focused_indices:
        sample_count = min(6, y.shape[0])
        print("Focused VSC control first samples:")
        for k in range(sample_count):
            values = ", ".join(f"{name}={y[k, idx]: .6e}" for name, idx in focused_indices)
            print(f"  k={k}, t={float(t[k]):.9e}: {values}")

    state_vars = problem.get_state_vars()
    alg_vars = problem.get_algebraic_vars()
    all_vars = state_vars + alg_vars
    seed_to_last_rows = []
    seed_to_first_rows = []
    seed_to_last_no_angle_rows = []
    wrapped_angle_rows = []
    unseeded_rows = []
    for var in all_vars:
        idx = problem._uid2idx_vars.get(var.uid, None)
        if idx is None:
            continue
        delta_last = float(x_last[idx] - x0_seed[idx])
        delta_first = float(x_first[idx] - x0_seed[idx])
        if abs(delta_last) > 1.0e-10:
            seed_to_last_rows.append((abs(delta_last), f"{var.name}: seed={x0_seed[idx]: .9e}, xlast={x_last[idx]: .9e}, delta={delta_last: .9e}"))
            if "theta" in var.name.lower() or "angle" in var.name.lower():
                wrapped_delta = wrapped_angle_delta(float(x_last[idx]), float(x0_seed[idx]))
                wrapped_angle_rows.append((abs(wrapped_delta), f"{var.name}: seed={x0_seed[idx]: .9e}, xlast={x_last[idx]: .9e}, raw_delta={delta_last: .9e}, wrapped_delta={wrapped_delta: .9e}"))
            else:
                seed_to_last_no_angle_rows.append((abs(delta_last), f"{var.name}: seed={x0_seed[idx]: .9e}, xlast={x_last[idx]: .9e}, delta={delta_last: .9e}"))
        if abs(delta_first) > 1.0e-10:
            seed_to_first_rows.append((abs(delta_first), f"{var.name}: seed={x0_seed[idx]: .9e}, x1={x_first[idx]: .9e}, delta={delta_first: .9e}"))
        if var.uid not in problem.init_guess and abs(x_last[idx]) > 1.0e-10:
            unseeded_rows.append((abs(x_last[idx]), f"{var.name}: seed-default=0, xlast={x_last[idx]: .9e}"))

    print_ranked_diagnostics("Largest xlast - seeded x0 differences:", seed_to_last_rows)
    print_ranked_diagnostics("Largest xlast - seeded x0 differences excluding angles:", seed_to_last_no_angle_rows)
    print_ranked_diagnostics("Wrapped angle xlast - seeded x0 differences:", wrapped_angle_rows)
    print_ranked_diagnostics("Largest first-step x1 - seeded x0 differences:", seed_to_first_rows)
    print_ranked_diagnostics("Unseeded variables that settle non-zero:", unseeded_rows)

    rhs0 = evaluate_state_rhs(problem, x0_seed, dx0_seed, t0)
    rhs_first = evaluate_state_rhs(problem, x_first, dx_first, t_first)
    rhs_last = evaluate_state_rhs(problem, x_last, dx_last, t_last)
    uid2idx_diff = getattr(problem, "uid2idx_diff", getattr(problem, "_uid2idx_diff", {}))
    base_uid_to_diff_var = {
        diff_var.base_var.uid: diff_var
        for diff_var in problem.get_diff_vars()
        if getattr(diff_var, "base_var", None) is not None
    }
    dx_seed_rows = []
    dx_last_delta_rows = []
    dx_first_delta_rows = []
    dx_first_rows = []
    dx_last_rows = []
    missing_dx_rows = []
    for i, var in enumerate(state_vars):
        diff_var = getattr(var, "diff_var", None)
        if diff_var is not None and diff_var.uid not in uid2idx_diff:
            diff_var = None
        if diff_var is None:
            diff_var = base_uid_to_diff_var.get(var.uid, None)
        if diff_var is None and i < len(problem.get_diff_vars()):
            # Some assembled/copied blocks lose the back-reference from state Var
            # to DiffVar. The solver still stores dx in state-equation order.
            diff_var = problem.get_diff_vars()[i]
        diff_idx = None if diff_var is None else uid2idx_diff.get(diff_var.uid, None)
        if diff_idx is None:
            continue

        seed_residual = float(dx0_seed[diff_idx] - rhs0[i])
        first_residual = float(dx_first[diff_idx] - rhs_first[i])
        last_residual = float(dx_last[diff_idx] - rhs_last[i])
        dx_first_delta = float(dx_first[diff_idx] - dx0_seed[diff_idx])
        dx_last_delta = float(dx_last[diff_idx] - dx0_seed[diff_idx])
        if abs(dx_first_delta) > 1.0e-10:
            dx_first_delta_rows.append((abs(dx_first_delta), f"{var.name}/{diff_var.name}: dx0={dx0_seed[diff_idx]: .9e}, dx1={dx_first[diff_idx]: .9e}, delta={dx_first_delta: .9e}"))
        if abs(dx_last_delta) > 1.0e-10:
            dx_last_delta_rows.append((abs(dx_last_delta), f"{var.name}/{diff_var.name}: dx0={dx0_seed[diff_idx]: .9e}, dxlast={dx_last[diff_idx]: .9e}, delta={dx_last_delta: .9e}"))
        dx_seed_rows.append((abs(seed_residual), f"{var.name}/{diff_var.name}: dx0={dx0_seed[diff_idx]: .9e}, rhs0={rhs0[i]: .9e}, dx0-rhs0={seed_residual: .9e}"))
        dx_first_rows.append((abs(first_residual), f"{var.name}/{diff_var.name}: dx1={dx_first[diff_idx]: .9e}, rhs1={rhs_first[i]: .9e}, dx1-rhs1={first_residual: .9e}"))
        dx_last_rows.append((abs(last_residual), f"{var.name}/{diff_var.name}: dxlast={dx_last[diff_idx]: .9e}, rhslast={rhs_last[i]: .9e}, dxlast-rhs={last_residual: .9e}"))
        if diff_var.uid not in problem.diff_init_guess and abs(rhs0[i]) > 1.0e-10:
            missing_dx_rows.append((abs(rhs0[i]), f"{var.name}/{diff_var.name}: dx0-default=0, rhs0={rhs0[i]: .9e}"))

    print_ranked_diagnostics("Largest dxlast - seeded dx0 differences:", dx_last_delta_rows)
    print_ranked_diagnostics("Largest first-step dx1 - seeded dx0 differences:", dx_first_delta_rows)
    print_ranked_diagnostics("Largest seeded dx0 - RHS(x0) mismatches:", dx_seed_rows)
    print_ranked_diagnostics("Largest accepted first-step dx1 - RHS(x1) mismatches:", dx_first_rows)
    print_ranked_diagnostics("Largest final dxlast - RHS(xlast) mismatches:", dx_last_rows)
    print_ranked_diagnostics("Unseeded differential variables with non-zero RHS at x0:", missing_dx_rows)

    alg0 = evaluate_algebraic_residuals(problem, x0_seed, dx0_seed, t0)
    alg_last = evaluate_algebraic_residuals(problem, x_last, dx_last, t_last)
    alg_rows0 = []
    alg_rows_last = []
    for i, eq in enumerate(problem.get_algebraic_eqs()):
        alg_rows0.append((abs(float(alg0[i])), f"alg_eq[{i}]: residual0={alg0[i]: .9e}, eq={eq}"))
        alg_rows_last.append((abs(float(alg_last[i])), f"alg_eq[{i}]: residuallast={alg_last[i]: .9e}, eq={eq}"))
    print_ranked_diagnostics("Largest algebraic residuals at seeded x0:", alg_rows0)
    print_ranked_diagnostics("Largest algebraic residuals at xlast:", alg_rows_last)


def apply_solution_as_initial_guess(problem: EmtProblemDae, y: np.ndarray, dy: np.ndarray) -> None:
    final_y = np.asarray(y[-1, :], dtype=float)
    for var in list(problem.get_state_vars()) + list(problem.get_algebraic_vars()):
        if var.uid in problem._uid2idx_vars:
            problem.init_guess[var.uid] = float(final_y[problem.get_var_idx(var)])

    uid2idx_diff = getattr(problem, "uid2idx_diff", getattr(problem, "_uid2idx_diff", {}))
    if dy is None or len(uid2idx_diff) == 0:
        return

    final_dy = np.asarray(dy[-1, :], dtype=float)
    for diff_var in problem.get_diff_vars():
        idx = uid2idx_diff.get(diff_var.uid, None)
        if idx is not None and idx < final_dy.size:
            problem.diff_init_guess[diff_var.uid] = float(final_dy[idx])


def run_emt(problem: EmtProblemDae, options: EmtOptions, t_end: float):
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=t_end,
        h=options.time_step,
        method=options.integration_method,
        newton_diag_config=NewtonDiagnosticsConfig(
            enable_fallback=True,
            enable_backtracking=True,
            backtracking_min_alpha=1.0e-6,
            backtracking_max_iter=12,
        ),
    )
    return solver.simulate()


def run_post_seed_initialization_if_requested(problem: EmtProblemDae, options: EmtOptions) -> None:
    if not POST_SEED_INITIALIZATION_METHOD_NAME:
        return

    method = EmtInitializationMethod[POST_SEED_INITIALIZATION_METHOD_NAME]
    previous_method = options.initialization_method
    options.initialization_method = method
    report = run_emt_native_initialization(problem=problem, options=options)
    options.initialization_method = previous_method
    print(
        "Post-seed native initialization: "
        f"requested={method.name}, used={report.method_used.name}, status={report.status.name}, "
        f"res0={report.initial_residual_inf:.9e}, resf={report.final_residual_inf:.9e}, "
        f"newton={report.newton_iterations}, ptc={report.pseudo_transient_steps}"
    )


def add_internal_filter_block(grid: gce.MultiCircuit, vsc_mdl: Block) -> None:
    vf = grid.var_factory
    r_filter = vf.add_var('R_filter')
    theta = find_name_in_block('theta', vsc_mdl)
    v_d_c = find_name_in_block('v_d_c', vsc_mdl)
    v_q_c = find_name_in_block('v_q_c', vsc_mdl)
    y_vd_hat = find_name_in_block('y_vd_hat', vsc_mdl)
    y_vq_hat = find_name_in_block('y_vq_hat', vsc_mdl)
    v_d_g = find_name_in_block('vg_d', vsc_mdl)
    v_q_g = find_name_in_block('vg_q', vsc_mdl)
    p_meas = find_name_in_block('P', vsc_mdl)
    q_meas = find_name_in_block('Q', vsc_mdl)
    l_filter = find_name_in_block('L', vsc_mdl)
    omega = find_name_in_block('omega', vsc_mdl)
    i_d = find_name_in_block('i_line_d', vsc_mdl)
    i_q = find_name_in_block('i_line_q', vsc_mdl)
    i_d_ref = find_name_in_block('i_d_ref', vsc_mdl)
    i_q_ref = find_name_in_block('i_q_ref', vsc_mdl)
    u_vd_hat = find_name_in_block('u_vd_hat', vsc_mdl)
    u_vq_hat = find_name_in_block('u_vq_hat', vsc_mdl)
    park_theta = vf.add_const(0.0) - theta
    sqrt3 = vf.add_const(np.sqrt(3.0))
    half = vf.add_const(0.5)
    third = vf.add_const(1.0 / 3.0)
    va_conv = vf.add_var('va_v')
    vb_conv = vf.add_var('vb_v')
    vc_conv = vf.add_var('vc_v')
    i_filter_a = vf.add_var('i_filter_A')
    i_filter_b = vf.add_var('i_filter_B')
    i_filter_c = vf.add_var('i_filter_C')
    d_i_filter_a = vf.add_diff_var(name='dt_i_filter_A', base_var=i_filter_a)
    d_i_filter_b = vf.add_diff_var(name='dt_i_filter_B', base_var=i_filter_b)
    d_i_filter_c = vf.add_diff_var(name='dt_i_filter_C', base_var=i_filter_c)
    omega_base = vf.add_const(2.0 * np.pi * grid.fBase)
    v_bus_a = find_name_in_block('vg_A', vsc_mdl)
    v_bus_b = find_name_in_block('vg_B', vsc_mdl)
    v_bus_c = find_name_in_block('vg_C', vsc_mdl)
    i_vsc_a = vsc_mdl.external_mapping[VarPowerFlowReferenceType.i_A]
    i_vsc_b = vsc_mdl.external_mapping[VarPowerFlowReferenceType.i_B]
    i_vsc_c = vsc_mdl.external_mapping[VarPowerFlowReferenceType.i_C]
    if not hasattr(i_vsc_a, 'non_mutable_uid'):
        i_vsc_a = find_name_in_block('i_filter_A', vsc_mdl)
    if not hasattr(i_vsc_b, 'non_mutable_uid'):
        i_vsc_b = find_name_in_block('i_filter_B', vsc_mdl)
    if not hasattr(i_vsc_c, 'non_mutable_uid'):
        i_vsc_c = find_name_in_block('i_filter_C', vsc_mdl)
    vsc_mdl.update_model(i_vsc_a, i_filter_a)
    vsc_mdl.update_model(i_vsc_b, i_filter_b)
    vsc_mdl.update_model(i_vsc_c, i_filter_c)
    filter_current_uids = {i_filter_a.uid, i_filter_b.uid, i_filter_c.uid}

    for block in vsc_mdl.get_all_blocks():
        filtered_eqs = []
        for eq in block.algebraic_eqs:
            eq_text = str(eq)
            if u_vd_hat is not None and eq_text.startswith("(u_vd_hat) - "):
                filtered_eqs.append(u_vd_hat - (i_d - i_d_ref))
                continue
            if u_vq_hat is not None and eq_text.startswith("(u_vq_hat) - "):
                filtered_eqs.append(u_vq_hat - (i_q - i_q_ref))
                continue
            if p_meas is not None and eq_text.startswith("(P) - ((0.5)"):
                continue
            if q_meas is not None and eq_text.startswith("(Q) - ((0.5)"):
                continue
            if not (
                str(eq).startswith("(v_d_c) - (((y_vd_hat) + (vg_d))")
                or str(eq).startswith("(v_q_c) - (((y_vq_hat) + (vg_q))")
                or str(eq) == "(vc_d) - (v_d_c)"
                or str(eq) == "(vc_q) - (v_q_c)"
                or str(eq) == "((vc_A) + (vc_B)) + (vc_C)"
                or str(eq).startswith("(vc_d) - ((0.3333333333333333)")
                or str(eq).startswith("(vc_q) - ((0.3333333333333333)")
            ):
                filtered_eqs.append(eq)
        block.algebraic_eqs = filtered_eqs
        block.algebraic_vars = [
            var for var in block.algebraic_vars
            if hasattr(var, 'non_mutable_uid')
            and getattr(var, 'name', None) not in {'vc_d', 'vc_q'}
            and getattr(var, 'uid', None) not in filter_current_uids
        ]

    filter_block = Block(
        algebraic_vars=[va_conv, vb_conv, vc_conv],
        state_vars=[i_filter_a, i_filter_b, i_filter_c],
        diff_vars=[d_i_filter_a, d_i_filter_b, d_i_filter_c],
        state_eqs=[
            omega_base * (v_bus_a - va_conv - r_filter * i_filter_a) / l_filter,
            omega_base * (v_bus_b - vb_conv - r_filter * i_filter_b) / l_filter,
            omega_base * (v_bus_c - vc_conv - r_filter * i_filter_c) / l_filter,
        ],
        algebraic_eqs=[
            p_meas - third * (v_bus_a * i_filter_a + v_bus_b * i_filter_b + v_bus_c * i_filter_c),
            q_meas - (third / sqrt3) * (
                (v_bus_b - v_bus_c) * i_filter_a
                + (v_bus_c - v_bus_a) * i_filter_b
                + (v_bus_a - v_bus_b) * i_filter_c
            ),
            v_d_c - (y_vd_hat + v_d_g + r_filter * i_d - l_filter * omega * i_q),
            v_q_c - (y_vq_hat + v_q_g + r_filter * i_q + l_filter * omega * i_d),
            va_conv - (v_d_c * sym.cos(park_theta) + v_q_c * sym.sin(park_theta)),
            vb_conv - (v_d_c * (-half * sym.cos(park_theta) - half * sqrt3 * sym.sin(park_theta))
                       + v_q_c * (-half * sym.sin(park_theta) + half * sqrt3 * sym.cos(park_theta))),
            vc_conv - (v_d_c * (-half * sym.cos(park_theta) + half * sqrt3 * sym.sin(park_theta))
                       + v_q_c * (-half * sym.sin(park_theta) - half * sqrt3 * sym.cos(park_theta))),
        ],
        event_dict={
            r_filter: vf.add_const(0.0),
        },
        name="internal_vsc_filter_rl",
    )
    vsc_mdl.add(filter_block)
    vsc_mdl.unify_blocks()


def connect_internal_filter_ports(grid: gce.MultiCircuit, vsc_mdl: Block) -> None:
    def input_port(port_name: str):
        found = next((var for var in vsc_mdl.in_vars if getattr(var, 'name', None) == port_name), None)
        if found is not None:
            return found
        ref = {
            'vc_A': VarPowerFlowReferenceType.v_A,
            'vc_B': VarPowerFlowReferenceType.v_B,
            'vc_C': VarPowerFlowReferenceType.v_C,
        }.get(port_name)
        return vsc_mdl.external_mapping.get(ref, None) if ref is not None else None

    va_conv = find_name_in_block('va_v', vsc_mdl)
    vb_conv = find_name_in_block('vb_v', vsc_mdl)
    vc_conv = find_name_in_block('vc_v', vsc_mdl)
    i_vsc_a = vsc_mdl.external_mapping[VarPowerFlowReferenceType.i_A]
    i_vsc_b = vsc_mdl.external_mapping[VarPowerFlowReferenceType.i_B]
    i_vsc_c = vsc_mdl.external_mapping[VarPowerFlowReferenceType.i_C]
    if not hasattr(i_vsc_a, 'non_mutable_uid'):
        i_vsc_a = find_name_in_block('i_filter_A', vsc_mdl)
    if not hasattr(i_vsc_b, 'non_mutable_uid'):
        i_vsc_b = find_name_in_block('i_filter_B', vsc_mdl)
    if not hasattr(i_vsc_c, 'non_mutable_uid'):
        i_vsc_c = find_name_in_block('i_filter_C', vsc_mdl)

    grid.var_factory.add_connections(
        [input_port('vc_A'), input_port('vc_B'), input_port('vc_C')],
        [va_conv, vb_conv, vc_conv],
    )
    grid.var_factory.add_connections(
        [find_name_in_block('i_line_A', vsc_mdl), find_name_in_block('i_line_B', vsc_mdl), find_name_in_block('i_line_C', vsc_mdl)],
        [i_vsc_a, i_vsc_b, i_vsc_c],
    )


def add_bus_zero_sequence_reference(parent_block: Block, bus: gce.Bus) -> None:
    v_a = find_name_in_block('v_A', bus.emt_model)
    v_b = find_name_in_block('v_B', bus.emt_model)
    v_c = find_name_in_block('v_C', bus.emt_model)
    parent_block.add(
        Block(
            algebraic_eqs=[v_a + v_b + v_c],
            name=f"{bus.name}_zero_sequence_reference",
        )
    )


def build_grid():
    grid = gce.MultiCircuit(Sbase=100, fbase=50.0)

    bus_dc = gce.Bus(name='Bdc', Vnom=100, is_dc=True)
    bus2 = gce.Bus(name='B2', Vnom=135)
    bus3 = gce.Bus(name='B3', Vnom=135, is_slack=True)

    for bus in (bus_dc, bus2, bus3):
        grid.add_bus(bus)
        get_bus_emt_template(grid, bus)

    line23 = gce.Line(
        name="line 2-3",
        bus_from=bus2,
        bus_to=bus3,
        r=3.0 * 0.029585798816568046,
        x=3.0 * 0.07100591715976332,
        b=0.03 / 3.0,
        rate=900.0,
    )

    vsc1 = gce.VSC(
        name='VSC1',
        bus_from=bus_dc,
        bus_to=bus2,
        rate=100,
        alpha1=0.0,
        alpha2=0.0,
        alpha3=0.0,
        control1=ConverterControlType.Vm_dc,
        control2=ConverterControlType.Qac,
        control1_val=1.03,
        control2_val=1.001,
    )
    gen1 = gce.Generator(name='Gen1', P=5.0, vset=1.00, Snom=grid.Sbase, freq=50.0, r1=0.001, x1=1.7)
    dc_source = gce.Generator(name='DC Source', P=3.0, vset=1.03)

    grid.add_line(line23)
    grid.add_vsc(vsc1)
    grid.add_generator(bus=bus3, api_obj=gen1)
    grid.add_generator(bus=bus_dc, api_obj=dc_source)

    return grid, bus_dc, bus2, bus3, line23, vsc1, gen1, dc_source


def run_power_flow(grid: gce.MultiCircuit):
    options = gce.PowerFlowOptions(
        solver_type=gce.SolverType.NR,
        retry_with_other_methods=False,
        verbose=0,
        initialize_with_existing_solution=True,
        tolerance=1e-6,
        max_iter=25,
        control_q=False,
        control_taps_modules=True,
        control_taps_phase=True,
        control_remote_voltage=True,
        orthogonalize_controls=True,
        apply_temperature_correction=True,
        branch_impedance_tolerance_mode=gce.BranchImpedanceMode.Specified,
        distributed_slack=False,
        ignore_single_node_islands=False,
        trust_radius=1.0,
        backtracking_parameter=0.05,
        use_stored_guess=False,
        initialize_angles=False,
        generate_report=False,
    )
    return gce.power_flow(grid, options=options)


def build_emt_models(grid, line23, vsc1, gen1, dc_source, pf_results, bus_dc, bus2, bus3):
    line23_mdl = get_pi_line_emt_template(vf=grid.var_factory, name="Pi23").block
    gen_mdl = get_generator_thevenin_rl_emt_template_with_ref(grid.var_factory).block
    dc_source_mdl = get_dc_voltage_source_emt_template(
        vf=grid.var_factory,
        source_voltage_value=1.03,
        name="dc_voltage_source",
    ).block
    vsc_mdl = VscGflEmtBuild(
        grid.var_factory,
        control1=ConverterControlType.Pac,
        control2=ConverterControlType.Qac,
        frozen_voltage_source=False,
    ).block
    add_internal_filter_block(grid, vsc_mdl)

    set_emt_model(device=line23, model=line23_mdl, var_factory=grid.var_factory)
    set_emt_model(device=gen1, model=gen_mdl, var_factory=grid.var_factory)
    set_emt_model(device=dc_source, model=dc_source_mdl, var_factory=grid.var_factory)
    set_emt_model(device=vsc1, model=vsc_mdl, var_factory=grid.var_factory)
    connect_internal_filter_ports(grid, vsc_mdl)

    grid.var_factory.add_connections(
        [find_name_in_block('vg_A', vsc_mdl), find_name_in_block('vg_B', vsc_mdl), find_name_in_block('vg_C', vsc_mdl)],
        [find_name_in_block('v_A', bus2.emt_model), find_name_in_block('v_B', bus2.emt_model), find_name_in_block('v_C', bus2.emt_model)],
    )

    vsc_filter_r = 3.0 * 0.029585798816568046
    vsc_filter_l = 3.0 * 0.07100591715976332

    va_pcc0, vb_pcc0, vc_pcc0 = pf_voltage_abc(grid, pf_results, bus2)
    best_theta_ref = 0.0
    best_abs_d_ref = None
    for theta_candidate in np.linspace(-np.pi, np.pi, 40001):
        vd_candidate, vq_candidate = park_dq_value(theta_candidate, va_pcc0, vb_pcc0, vc_pcc0)
        if vq_candidate > 0.0 and (best_abs_d_ref is None or abs(vd_candidate) < best_abs_d_ref):
            best_abs_d_ref = abs(vd_candidate)
            best_theta_ref = theta_candidate

    line23_idx = grid.lines.index(line23)
    line23_s_from = complex(pf_results.Sf[line23_idx]) / grid.Sbase
    p_ref0 = -float(np.real(line23_s_from))
    q_ref0 = -float(np.imag(line23_s_from))
    vq_ref0 = park_dq_value(best_theta_ref, va_pcc0, vb_pcc0, vc_pcc0)[1]

    for name, value in (
        ('Vdc_ref', abs(pf_results.voltage[grid.buses.index(bus_dc)])),
        ('Vm_ac_ref', vq_ref0),
        ('P_ref', p_ref0),
        ('Q_ref', q_ref0),
        ('Kp_pol', Q_CONTROL_KP_POL),
        ('Ki_pol', Q_CONTROL_KI_POL),
        ('Kp_icl', CURRENT_CONTROL_KP_ICL),
        ('Ki_icl', CURRENT_CONTROL_KI_ICL),
        ('Kp_pll', PLL_KP),
        ('Ki_pll', PLL_KI),
        ('L', vsc_filter_l),
        ('Cdc', DC_LINK_CAPACITANCE),
        ('R_filter', vsc_filter_r),
        ('L_filter', vsc_filter_l),
    ):
        set_event_value_in_block(grid.var_factory, name, vsc_mdl, value)

    if P_REF_EVENT_ENABLED:
        p_ref_var = find_name_in_block('P_ref', vsc_mdl)
        if p_ref_var is None:
            raise RuntimeError("Cannot add P_ref event: P_ref was not found in the VSC EMT model")
        p_ref_event_value = p_ref0 + P_REF_EVENT_DELTA
        grid.add_emt_event(EmtEvent(device=vsc1, parameter=p_ref_var, time=P_REF_EVENT_TIME, value=p_ref_event_value))
        print(
            "Added P_ref EMT event: "
            f"t={P_REF_EVENT_TIME:g}, initial={p_ref0:.9e}, "
            f"value={p_ref_event_value:.9e}, delta={P_REF_EVENT_DELTA:.9e}"
        )

    return vsc_mdl, line23_mdl, gen_mdl, dc_source_mdl, vsc_filter_l, p_ref0, q_ref0, best_theta_ref


def seed_internal_filter_guess(problem, grid, pf_results, vsc_mdl, dc_source_mdl, line23, bus_dc, bus2, theta_ref):
    va2, vb2, vc2 = pf_voltage_abc(grid, pf_results, bus2)
    vd_g, vq_g = park_dq_value(theta_ref, va2, vb2, vc2)

    line23_idx = grid.lines.index(line23)
    s_to = complex(pf_results.Sf[line23_idx]) / grid.Sbase
    i_q = 2.0 * s_to.real / vq_g
    i_d = 2.0 * s_to.imag / vq_g
    r_filter = 3.0 * 0.029585798816568046
    x_filter = 3.0 * 0.07100591715976332
    vd_c = vd_g + r_filter * i_d - x_filter * i_q
    vq_c = vq_g + r_filter * i_q + x_filter * i_d
    va_conv, vb_conv, vc_conv = inverse_park_value(theta_ref, vd_c, vq_c)
    ia, ib, ic = inverse_park_value(theta_ref, i_d, i_q)
    ia_filter = -ia
    ib_filter = -ib
    ic_filter = -ic
    i_d_filter = -i_d
    i_q_filter = -i_q
    vdc0 = abs(pf_results.voltage[grid.buses.index(bus_dc)])
    p_meas = (va2 * ia_filter + vb2 * ib_filter + vc2 * ic_filter) / 3.0
    q_meas = ((vb2 - vc2) * ia_filter + (vc2 - va2) * ib_filter + (va2 - vb2) * ic_filter) / (3.0 * np.sqrt(3.0))
    p_conv = -(va_conv * ia_filter + vb_conv * ib_filter + vc_conv * ic_filter) / 3.0
    omega_base = 2.0 * np.pi * grid.fBase
    dia = omega_base * (va2 - va_conv - r_filter * ia_filter) / x_filter
    dib = omega_base * (vb2 - vb_conv - r_filter * ib_filter) / x_filter
    dic = omega_base * (vc2 - vc_conv - r_filter * ic_filter) / x_filter
    vd_hat = vd_c - (vd_g + r_filter * i_d_filter - x_filter * i_q_filter)
    vq_hat = vq_c - (vq_g + r_filter * i_q_filter + x_filter * i_d_filter)

    values = {
        'P': p_meas,
        'Q': q_meas,
        'P_f': p_meas,
        'Q_f': q_meas,
        'Pt_vsc': -p_meas,
        'Qt_vsc': -q_meas,
        'i_a_f': ia_filter,
        'i_b_f': ib_filter,
        'i_c_f': ic_filter,
        'i_filter_A': ia_filter,
        'i_filter_B': ib_filter,
        'i_filter_C': ic_filter,
        'i_dc': p_conv / vdc0,
        'P_conv': p_conv,
        'theta': -theta_ref,
        'omega': 1.0,
        'xi_PLL': 0.0,
        'vg_d': vd_g,
        'vg_q': vq_g,
        'vc_d': vd_c,
        'vc_q': vq_c,
        'i_line_d': i_d_filter,
        'i_line_q': i_q_filter,
        'i_d_ref': i_d_filter,
        'i_q_ref': i_q_filter,
        'v_d_c': vd_c,
        'v_q_c': vq_c,
        'va_v': va_conv,
        'vb_v': vb_conv,
        'vc_v': vc_conv,
        'u_Pac_ctrl': 0.0,
        'u_Qac_ctrl': 0.0,
        'y_vd_hat': vd_hat,
        'y_vq_hat': vq_hat,
        'u_vd_hat': 0.0,
        'u_vq_hat': 0.0,
        'Vdc_cap': vdc0,
        'xi_Pac_ctrl': i_q_filter / Q_CONTROL_KI_POL,
        'xi_Qac_ctrl': i_d_filter / Q_CONTROL_KI_POL,
        'xi_vd_hat': vd_hat / CURRENT_CONTROL_KI_ICL,
        'xi_vq_hat': vq_hat / CURRENT_CONTROL_KI_ICL,
        'u_PLL_pi': vd_g,
    }
    for name, value in values.items():
        set_init_if_exists(problem, vsc_mdl, name, value)

    dc_current = p_conv / vdc0
    for name, value in (('i_dc', dc_current), ('i_dc_dc_voltage_source', dc_current)):
        var = find_name_in_block(name, dc_source_mdl)
        if var is not None:
            if var.uid in problem._uid2idx_vars:
                problem.init_guess[var.uid] = float(value)
            elif var.uid in problem.uid2idx_event_params:
                problem.event_params_values[problem.uid2idx_event_params[var.uid]] = float(value)

    set_problem_runtime_value(problem, grid.var_factory, dc_source_mdl, 'V_src', vdc0 + dc_current / 100.0)

    for name, value in (
        ('P_ref', p_meas),
        ('Q_ref', q_meas),
        ('Vm_ac_ref', vq_g),
        ('Kp_pol', Q_CONTROL_KP_POL),
        ('Ki_pol', Q_CONTROL_KI_POL),
        ('Kp_icl', CURRENT_CONTROL_KP_ICL),
        ('Ki_icl', CURRENT_CONTROL_KI_ICL),
        ('Kp_pll', PLL_KP),
        ('Ki_pll', PLL_KI),
        ('L', x_filter),
        ('Cdc', DC_LINK_CAPACITANCE),
    ):
        set_problem_runtime_value(problem, grid.var_factory, vsc_mdl, name, value)

    theta_var = find_name_in_block('theta', vsc_mdl)
    if theta_var is not None and theta_var.diff_var is not None:
        problem.diff_init_guess[theta_var.diff_var.uid] = 2.0 * np.pi * grid.fBase
    vdc_cap_var = find_name_in_block('Vdc_cap', vsc_mdl)
    if vdc_cap_var is not None and vdc_cap_var.diff_var is not None:
        problem.diff_init_guess[vdc_cap_var.diff_var.uid] = 0.0
    for name, value in (('dt_i_filter_A', dia), ('dt_i_filter_B', dib), ('dt_i_filter_C', dic)):
        diff_var = find_name_in_block(name, vsc_mdl)
        if diff_var is not None:
            problem.diff_init_guess[diff_var.uid] = float(value)

    # The active/reactive power PI blocks expose derivative variables for both
    # the error signal and the reference output. At the PF operating point the
    # errors are initialized to zero, so their startup derivatives should also be
    # zero. Leaving them to automatic completion can make i_q_ref/i_d_ref jump on
    # the first trapezoidal step even though i_ref == i at t=0.
    for name in ('dt_1_i_q_ref', 'dt_1_u_Pac_ctrl', 'dt_1_i_d_ref', 'dt_1_u_Qac_ctrl'):
        diff_var = find_name_in_block(name, vsc_mdl)
        if diff_var is not None:
            problem.diff_init_guess[diff_var.uid] = 0.0

    # The GFL PI blocks are represented as algebraic equations involving
    # derivative variables, e.g. dt_i_q_ref = Ki * u_Pac_ctrl + Kp * dt_u_Pac_ctrl.
    # At the seeded operating point the power errors are zero, so keep both the
    # PI input derivative and output derivative at zero to avoid a first-step
    # trapezoidal predictor kick in i_q_ref/i_d_ref.
    for name in ('dt_1_i_q_ref', 'dt_1_i_d_ref', 'dt_1_u_Pac_ctrl', 'dt_1_u_Qac_ctrl'):
        diff_var = find_name_in_block(name, vsc_mdl)
        if diff_var is not None:
            problem.diff_init_guess[diff_var.uid] = 0.0


def main() -> None:
    grid, bus_dc, bus2, bus3, line23, vsc1, gen1, dc_source = build_grid()
    pf_results = run_power_flow(grid)
    print(f"Power flow converged: {pf_results.converged}")
    print(pf_results.get_bus_df())
    print(pf_results.get_branch_df())

    vsc_mdl, line23_mdl, gen_mdl, dc_source_mdl, vsc_filter_l, p_ref0, q_ref0, theta_ref = build_emt_models(
        grid, line23, vsc1, gen1, dc_source, pf_results, bus_dc, bus2, bus3
    )
    print("Topology: VSC is connected to B2; only line 2-3 is in the MultiCircuit")
    print(f"Internal VSC filter L from former line 1-2 X: {vsc_filter_l:.9e}")
    print(f"Initial internal-filter references: P_ref={p_ref0:.9e}, Q_ref={q_ref0:.9e}")

    options = EmtOptions(
        time_step=VERAGRID_GFL_EMT_TIME_STEP,
        simulation_time=VERAGRID_GFL_EMT_SIM_TIME,
        tolerance=1e-6,
        integration_method=DynamicIntegrationMethod[_DEFAULT_INTEGRATION_METHOD_NAME],
        solver_type=EmtSolverTypes.StructuralAD,
        initialization_method=EmtInitializationMethod.Explicit,
        verbose=0,
    )
    problem = EmtProblemDae(grid=grid, options=options, pf_results=pf_results)
    seed_internal_filter_guess(problem, grid, pf_results, vsc_mdl, dc_source_mdl, line23, bus_dc, bus2, theta_ref)
    run_post_seed_initialization_if_requested(problem, options)
    print_problem_structure(problem)
    print_vsc_seed_values(problem, vsc_mdl)

    try:
        if VERAGRID_GFL_EMT_WARMUP_TIME > 0.0:
            print(f"Running EMT warm-up: {VERAGRID_GFL_EMT_WARMUP_TIME:.6g} s")
            tw, yw, dyw, warm_well_initialized, warm_converged = run_emt(
                problem=problem,
                options=options,
                t_end=VERAGRID_GFL_EMT_WARMUP_TIME,
            )
            print(f"EMT warm-up well initialized: {warm_well_initialized}")
            print(f"EMT warm-up converged: {warm_converged}")
            apply_solution_as_initial_guess(problem, yw, dyw)

        x0_run = problem.get_x0().copy()
        dx0_run = problem.get_dx0().copy()
        t, y, dy, well_initialized, converged = run_emt(
            problem=problem,
            options=options,
            t_end=options.simulation_time,
        )
    except np.linalg.LinAlgError as exc:
        print(f"EMT solve failed during Newton iteration: {exc}")
        print("The assembled EMT system is structurally square, but the initial Jacobian is singular.")
        return
    print(f"EMT well initialized: {well_initialized}")
    print(f"EMT converged: {converged}")
    print(f"Solved samples: {len(t)}")
    print_initialization_diagnostics(problem, vsc_mdl, t, y, dy, x0_run, dx0_run)

    for name in ('P', 'Q', 'v_d_c', 'v_q_c', 'vg_d', 'vg_q', 'i_line_d', 'i_line_q'):
        var = find_name_in_block(name, vsc_mdl)
        if var is not None and var.uid in problem._uid2idx_vars:
            trace = y[:, problem.get_var_idx(var)]
            print(f"{name}: t0={trace[0]: .9e}, final={trace[-1]: .9e}")

    plot_traces(t, problem, y, vsc_mdl, bus_dc, bus2, p_ref0)
    print(f"Plots saved to: {PLOT_OUTPUT_DIR}")


if __name__ == '__main__':
    main()
