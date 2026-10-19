#!/usr/bin/env python3
"""Validate the combined EMT GFM source model."""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import VeraGridEngine.api as vge
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Templates.Emt.load_zip_emt_template import get_load_ZIP_emt_template
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.balanced_source_emt_template import get_balanced_3ph_voltage_source_emt_template
from VeraGridEngine.Templates.Emt.simple_generator_emt_template import get_simple_generator_emt_template
from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import get_generator_thevenin_rl_emt_template_with_ref
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.diagnostic import NewtonDiagnosticsConfig
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes

from TensyGridEngine.emt.validation_helpers.emt_gfm_model import (
    build_emt_gfm_model, create_grid, find_name_in_block,
)


PLOT_OUTPUT_DIR = os.environ.get(
    "VERAGRID_GFM_EMT_PLOT_OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "emt_gfm_model_validation_plots"),
)
TIME_STEP = float(os.environ.get("VERAGRID_GFM_EMT_TIME_STEP", "1e-5"))
SIM_TIME = float(os.environ.get("VERAGRID_GFM_EMT_SIM_TIME", "0.1"))
WARMUP_TIME = float(os.environ.get("VERAGRID_GFM_EMT_WARMUP_TIME", "0.0"))
METHOD_NAME = os.environ.get("VERAGRID_GFM_EMT_METHOD", "trap").strip().lower()
SLACK_SOURCE = os.environ.get("VERAGRID_GFM_EMT_SLACK_SOURCE", "thevenin").strip().lower()
LOAD_SCALE = float(os.environ.get("VERAGRID_GFM_EMT_LOAD_SCALE", "1.0"))


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


P_REF_STEP = os.environ.get("VERAGRID_GFM_EMT_P_REF_STEP", None)
P_REF_STEP_TIME = float(os.environ.get("VERAGRID_GFM_EMT_P_REF_STEP_TIME", "0.1"))


def integration_method_from_env() -> DynamicIntegrationMethod:
    if METHOD_NAME in {"trap", "trapezoidal", "dae_trapezoidal"}:
        return DynamicIntegrationMethod.DaeTrapezoidal
    return DynamicIntegrationMethod.DaeBackEuler


def run_power_flow(grid: vge.MultiCircuit):
    options = vge.PowerFlowOptions(
        solver_type=vge.SolverType.NR,
        verbose=0,
        retry_with_other_methods=True,
        max_iter=50,
        tolerance=1.0e-8,
        control_q=False,
        control_taps_modules=False,
        control_taps_phase=False,
        orthogonalize_controls=False,
    )
    return vge.power_flow(grid=grid, options=options)


def create_simplified_grid() -> tuple[vge.MultiCircuit, dict]:
    """Create a minimal slack-line-GFM EMT validation case."""

    grid = vge.MultiCircuit(name="EMT GFM simplified validation", Sbase=100.0, fbase=50.0)

    bus_slack = vge.Bus(name="Bus 1 (AC)", Vnom=230.0, is_slack=True)
    bus_gfm = vge.Bus(name="Bus 1_1 (AC)", Vnom=230.0)
    grid.add_bus(bus_slack)
    grid.add_bus(bus_gfm)

    gen_slack = vge.Generator(name="Generator", P=0.0, vset=0.98, Snom=grid.Sbase, freq=50.0)
    gen_slack.R1 = float(os.environ.get("VERAGRID_GFM_EMT_THEVENIN_R", "0.01"))
    gen_slack.X1 = float(os.environ.get("VERAGRID_GFM_EMT_THEVENIN_X", "0.20"))
    gen_slack.X0 = gen_slack.X1
    gen_gfm = vge.Generator(name="GFM Generator", P=4.0, vset=0.98, Snom=grid.Sbase, freq=50.0)
    line13 = vge.Line(
        name="Line13",
        bus_from=bus_gfm,
        bus_to=bus_slack,
        rate=100.0,
        r=0.01,
        x=0.05,
        b=0.0,
    )

    grid.add_generator(bus=bus_slack, api_obj=gen_slack)
    grid.add_generator(bus=bus_gfm, api_obj=gen_gfm)
    grid.add_line(line13)
    grid.apply_all_branch_types()

    devices = {
        "bus_slack": bus_slack,
        "bus_gfm": bus_gfm,
        "gen_slack": gen_slack,
        "gen_gfm": gen_gfm,
        "line13": line13,
    }
    return grid, devices


def attach_emt_models(grid: vge.MultiCircuit, devices: dict, pf_results=None) -> object:
    for bus in grid.buses:
        get_bus_emt_template(grid=grid, bus=bus)

    gfm_block = build_emt_gfm_model(vf=grid.var_factory, name="GFM_combined")
    set_emt_model(devices["gen_gfm"], gfm_block, grid.var_factory)

    for line in grid.lines:
        line_block = get_pi_line_emt_template(vf=grid.var_factory, name=line.name.replace(" ", "_")).block
        set_emt_model(line, line_block, grid.var_factory)

    if "gen_slack" in devices:
        if SLACK_SOURCE in {"balanced", "voltage", "voltage_source", "norton"} and pf_results is not None:
            bus_idx = grid.buses.index(devices["bus_slack"])
            voltage = complex(pf_results.voltage[bus_idx])
            source_g = float(os.environ.get("VERAGRID_GFM_EMT_SLACK_G", "1000.0"))
            vpk = float(np.sqrt(2.0) * abs(voltage))
            angle = float(np.angle(voltage))
            p0 = float(np.real(pf_results.Sbus[bus_idx]) / grid.Sbase)
            q0 = float(np.imag(pf_results.Sbus[bus_idx]) / grid.Sbase)
            va0 = vpk * np.sin(angle)
            vb0 = vpk * np.sin(angle - 2.0 * np.pi / 3.0)
            vc0 = vpk * np.sin(angle + 2.0 * np.pi / 3.0)
            theta = angle

            def abc_from_dq(id_value: float, iq_value: float) -> tuple[float, float, float]:
                c = np.cos(theta)
                s = np.sin(theta)
                sqrt3 = np.sqrt(3.0)
                return (
                    id_value * c + iq_value * s,
                    id_value * (-0.5 * c + 0.5 * sqrt3 * s)
                    + iq_value * (-0.5 * s - 0.5 * sqrt3 * c),
                    id_value * (-0.5 * c - 0.5 * sqrt3 * s)
                    + iq_value * (-0.5 * s + 0.5 * sqrt3 * c),
                )

            def abc_power(id_value: float, iq_value: float) -> tuple[float, float]:
                ia0, ib0, ic0 = abc_from_dq(id_value, iq_value)
                p_value = (va0 * ia0 + vb0 * ib0 + vc0 * ic0) / 3.0
                q_value = ((vb0 - vc0) * ia0 + (vc0 - va0) * ib0 + (va0 - vb0) * ic0) / (3.0 * np.sqrt(3.0))
                return p_value, q_value

            pq_from_id = np.asarray(abc_power(1.0, 0.0), dtype=float)
            pq_from_iq = np.asarray(abc_power(0.0, 1.0), dtype=float)
            id0, iq0 = np.linalg.solve(np.column_stack([pq_from_id, pq_from_iq]), np.asarray([p0, q0], dtype=float))
            ia0, _, _ = abc_from_dq(float(id0), float(iq0))
            omega_base = 2.0 * np.pi * grid.fBase
            dva0 = omega_base * vpk * np.cos(angle)

            def phase_a_derivative(d_value: float, q_value: float) -> float:
                theta_dot = omega_base
                return theta_dot * (-d_value * np.sin(theta) + q_value * np.cos(theta))

            dia0 = phase_a_derivative(float(id0), float(iq0))
            vsrc_a0 = va0 + ia0 / source_g
            dvsrc_a0 = dva0 + dia0 / source_g
            source_amp = float(np.hypot(vsrc_a0, dvsrc_a0 / omega_base))
            source_angle = float(np.rad2deg(np.arctan2(vsrc_a0, dvsrc_a0 / omega_base)))
            slack_block = get_balanced_3ph_voltage_source_emt_template(
                vf=grid.var_factory,
                amplitude_value=source_amp,
                frequency_hz=float(grid.fBase),
                phase_a_deg=source_angle,
                source_conductance_value=source_g,
                name="slack_balanced_voltage_source",
            ).block
        elif SLACK_SOURCE in {"thevenin", "thevenin_rl"}:
            slack_block = get_generator_thevenin_rl_emt_template_with_ref(vf=grid.var_factory).block
        else:
            slack_block = get_simple_generator_emt_template(vf=grid.var_factory).block
        set_emt_model(devices["gen_slack"], slack_block, grid.var_factory)

    if "load" in devices:
        load_block = get_load_ZIP_emt_template(vf=grid.var_factory).block
        set_emt_model(devices["load"], load_block, grid.var_factory)

    return gfm_block


def run_emt(problem: EmtProblemDae, options: EmtOptions, t_end: float, x0=None, dx0=None, params0=None):
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
            compute_dense_cond=env_flag("VERAGRID_GFM_EMT_COMPUTE_DENSE_COND", "1"),
            backtracking_min_alpha=1.0e-6,
            backtracking_max_iter=12,
            max_newton_step_inf=float(os.environ.get("VERAGRID_GFM_EMT_MAX_NEWTON_STEP", "0.05")),
        ),
    )
    predictor = getattr(solver, "_predictor", None)
    predictor_derivatives = getattr(predictor, "algebraic_derivative_by_index", {}) if predictor is not None else {}
    if predictor_derivatives:
        print(f"Algebraic predictor derivatives active: {len(predictor_derivatives)}")
    return solver.simulate(x0=x0, dx0=dx0, params0=params0)


def set_runtime_param_in_vector(problem: EmtProblemDae, block, name: str, params: np.ndarray, value: float) -> None:
    var = find_name_in_block(name, block)
    if var is None:
        raise ValueError(f"Could not find runtime parameter {name}")
    uid2idx = getattr(problem, "_uid2idx_event_params", getattr(problem, "uid2idx_event_params", {}))
    idx = uid2idx.get(var.uid, None)
    if idx is None:
        raise ValueError(f"{name} is not an EMT runtime parameter")
    params[idx] = float(value)
    if hasattr(problem, "_event_params_values"):
        problem._event_params_values[idx] = float(value)
    if hasattr(problem, "event_params_values"):
        problem.event_params_values[idx] = float(value)
    if hasattr(problem, "_event_parameters_eqs"):
        problem._event_parameters_eqs[idx] = problem.grid.var_factory.add_const(float(value))
    if hasattr(problem, "_runtime_continuous_eqs") and idx < len(problem._runtime_continuous_eqs):
        problem._runtime_continuous_eqs[idx] = problem.grid.var_factory.add_const(float(value))


def run_emt_with_p_ref_step(problem: EmtProblemDae, options: EmtOptions, gfm_block, step_time: float, step_delta: float):
    step_time = max(0.0, min(float(step_time), float(options.simulation_time)))
    print(f"Running EMT simulation with P_ref step: t={step_time:.6g} s, delta={step_delta:.9e}")
    t1, y1, dy1, well1, conv1 = run_emt(problem, options, step_time)
    params = problem.event_params_values.copy()
    p_ref = trace_or_param(problem, y1, "P_ref", gfm_block)
    p_ref_before = float(p_ref[-1])
    set_runtime_param_in_vector(problem, gfm_block, "P_ref", params, p_ref_before + float(step_delta))
    print(f"P_ref step applied: {p_ref_before:.9e} -> {p_ref_before + float(step_delta):.9e}")

    x0 = np.asarray(y1[-1, :], dtype=float)
    dx0 = np.asarray(dy1[-1, :], dtype=float)
    t2, y2, dy2, well2, conv2 = run_emt(problem, options, options.simulation_time - step_time, x0=x0, dx0=dx0, params0=params)

    if t2.size > 1:
        t = np.concatenate([t1, step_time + t2[1:]])
        y = np.vstack([y1, y2[1:, :]])
        dy = np.vstack([dy1, dy2[1:, :]])
    else:
        t, y, dy = t1, y1, dy1
    return t, y, dy, bool(well1 and well2), bool(conv1 and conv2)


def apply_solution_as_initial_guess(problem: EmtProblemDae, y: np.ndarray, dy: np.ndarray) -> None:
    final_y = np.asarray(y[-1, :], dtype=float)
    for var in list(problem.get_state_vars()) + list(problem.get_algebraic_vars()):
        if var.uid in problem._uid2idx_vars:
            problem.init_guess[var.uid] = float(final_y[problem.get_var_idx(var)])

    uid2idx_diff = getattr(problem, "uid2idx_diff", getattr(problem, "_uid2idx_diff", {}))
    final_dy = np.asarray(dy[-1, :], dtype=float)
    for diff_var in problem.get_diff_vars():
        idx = uid2idx_diff.get(diff_var.uid, None)
        if idx is not None and idx < final_dy.size:
            problem.diff_init_guess[diff_var.uid] = float(final_dy[idx])


def set_runtime_value(problem: EmtProblemDae, block, name: str, value: float) -> None:
    var = find_name_in_block(name, block)
    if var is None:
        return
    if var.uid in problem._uid2idx_vars:
        problem.init_guess[var.uid] = float(value)
    uid2idx = getattr(problem, "_uid2idx_event_params", getattr(problem, "uid2idx_event_params", {}))
    if var.uid in uid2idx:
        idx = uid2idx[var.uid]
        if hasattr(problem, "_event_params_values"):
            problem._event_params_values[idx] = float(value)
        if hasattr(problem, "event_params_values"):
            problem.event_params_values[idx] = float(value)
        if hasattr(problem, "event_params_init_dict"):
            problem.event_params_init_dict[var.uid] = float(value)
            problem.event_params_init_dict[idx] = float(value)


def set_block_event_value(block, vf, name: str, value: float) -> None:
    var = find_name_in_block(name, block)
    if var is not None and var in block.event_dict:
        block.event_dict[var] = vf.add_const(float(value))


def apply_gfm_gain_overrides(gfm_block, vf) -> None:
    overrides = {
        "Kp_vcl": "VERAGRID_GFM_EMT_KP_VCL",
        "Ki_vcl": "VERAGRID_GFM_EMT_KI_VCL",
        "Kp_icl": "VERAGRID_GFM_EMT_KP_ICL",
        "Ki_icl": "VERAGRID_GFM_EMT_KI_ICL",
        "Kdp": "VERAGRID_GFM_EMT_KDP",
        "Kdq": "VERAGRID_GFM_EMT_KDQ",
        "tau_P": "VERAGRID_GFM_EMT_TAU_P",
        "tau_Q": "VERAGRID_GFM_EMT_TAU_Q",
    }
    for gain_name, env_name in overrides.items():
        env_value = os.environ.get(env_name, None)
        if env_value is None:
            continue
        value = float(env_value)
        set_block_event_value(gfm_block, vf, gain_name, value)
        print(f"Overrode {gain_name}: {value:.9e}")


def seed_gfm_from_power_flow(problem: EmtProblemDae, grid: vge.MultiCircuit, pf_results, gfm_block, bus_gfm) -> None:
    theta_name_var = find_name_in_block("theta", gfm_block)
    theta_name = getattr(theta_name_var, "name", "") if theta_name_var is not None else ""
    model_name = theta_name[len("theta_"):] if theta_name.startswith("theta_") else str(gfm_block.name)
    bus_idx = grid.buses.index(bus_gfm)
    voltage = complex(pf_results.voltage[bus_idx])
    vpk = float(np.sqrt(2.0) * abs(voltage))
    angle = float(np.angle(voltage))
    theta = angle
    p0 = float(np.real(pf_results.Sbus[bus_idx]) / grid.Sbase)
    q0 = float(np.imag(pf_results.Sbus[bus_idx]) / grid.Sbase)
    va0 = vpk * np.sin(angle)
    vb0 = vpk * np.sin(angle - 2.0 * np.pi / 3.0)
    vc0 = vpk * np.sin(angle + 2.0 * np.pi / 3.0)

    def abc_from_dq(id_value: float, iq_value: float) -> tuple[float, float, float]:
        c = np.cos(theta)
        s = np.sin(theta)
        sqrt3 = np.sqrt(3.0)
        return (
            id_value * c + iq_value * s,
            id_value * (-0.5 * c + 0.5 * sqrt3 * s)
            + iq_value * (-0.5 * s - 0.5 * sqrt3 * c),
            id_value * (-0.5 * c - 0.5 * sqrt3 * s)
            + iq_value * (-0.5 * s + 0.5 * sqrt3 * c),
        )

    def abc_power(id_value: float, iq_value: float) -> tuple[float, float]:
        ia0, ib0, ic0 = abc_from_dq(id_value, iq_value)
        p_value = (va0 * ia0 + vb0 * ib0 + vc0 * ic0) / 3.0
        q_value = ((vb0 - vc0) * ia0 + (vc0 - va0) * ib0 + (va0 - vb0) * ic0) / (3.0 * np.sqrt(3.0))
        return p_value, q_value

    pq_from_id = np.asarray(abc_power(1.0, 0.0), dtype=float)
    pq_from_iq = np.asarray(abc_power(0.0, 1.0), dtype=float)
    id0, iq0 = np.linalg.solve(np.column_stack([pq_from_id, pq_from_iq]), np.asarray([p0, q0], dtype=float))
    p_seed, q_seed = abc_power(float(id0), float(iq0))
    print(
        "GFM PF seed check: "
        f"P_pf={p0:.9e}, P_abc={p_seed:.9e}, dP={p_seed - p0:.3e}; "
        f"Q_pf={q0:.9e}, Q_abc={q_seed:.9e}, dQ={q_seed - q0:.3e}"
    )
    rf = 0.02
    lf = 0.15
    rc = 0.01
    lc = 0.1
    cf = 0.05
    rcap = 1.0e6
    kp_vcl = float(os.environ.get("VERAGRID_GFM_EMT_KP_VCL", "0.00075"))
    ki_vcl = float(os.environ.get("VERAGRID_GFM_EMT_KI_VCL", "0.2"))
    kp_icl = float(os.environ.get("VERAGRID_GFM_EMT_KP_ICL", "0.00075"))
    ki_icl = float(os.environ.get("VERAGRID_GFM_EMT_KI_ICL", "0.2"))
    if abs(ki_vcl) <= 1.0e-12 or abs(ki_icl) <= 1.0e-12:
        raise ValueError("GFM voltage- and current-loop integral gains must be non-zero for explicit PI-state seeding")
    omega0 = 1.0
    vd_g0 = 0.0
    vq_g0 = vpk
    # These steady-state dq drops are derived from the phase-domain EMT
    # state equations with constant dq quantities in the rotating frame.
    vd_f0 = vd_g0 + rc * id0 + omega0 * lc * iq0
    vq_f0 = vq_g0 + rc * iq0 - omega0 * lc * id0
    id_c0 = id0 + cf * omega0 * vq_f0 + vd_f0 / rcap
    iq_c0 = iq0 - cf * omega0 * vd_f0 + vq_f0 / rcap
    vd_c0 = vd_f0 + rf * id_c0 + omega0 * lf * iq_c0
    vq_c0 = vq_f0 + rf * iq_c0 - omega0 * lf * id_c0

    # The GFM voltage loop regulates the filter-capacitor voltage, not the bus
    # voltage. Rotate the controller frame so the seeded capacitor voltage has
    # vd_f=0; otherwise the voltage integrator immediately relaxes the PF point.
    delta = float(np.arctan2(vd_f0, vq_f0))
    theta = float(theta + delta)

    def rotate_pair(d_value: float, q_value: float) -> tuple[float, float]:
        c_delta = np.cos(delta)
        s_delta = np.sin(delta)
        return (
            float(d_value * c_delta - q_value * s_delta),
            float(d_value * s_delta + q_value * c_delta),
        )

    vd_g0, vq_g0 = rotate_pair(vd_g0, vq_g0)
    id0, iq0 = rotate_pair(float(id0), float(iq0))
    vd_f0, vq_f0 = rotate_pair(vd_f0, vq_f0)
    id_c0, iq_c0 = rotate_pair(id_c0, iq_c0)
    vd_c0, vq_c0 = rotate_pair(vd_c0, vq_c0)
    v_ref0 = vq_f0

    z_vd0 = (id_c0 - id0 - cf * omega0 * vq_f0 - kp_vcl * (0.0 - vd_f0)) / ki_vcl
    z_vq0 = (iq_c0 - iq0 + cf * omega0 * vd_f0 - kp_vcl * (v_ref0 - vq_f0)) / ki_vcl
    vd_ctrl0 = vd_c0 - vd_f0 - lf * omega0 * iq_c0
    vq_ctrl0 = vq_c0 - vq_f0 + lf * omega0 * id_c0
    ia0, ib0, ic0 = abc_from_dq(float(id0), float(iq0))
    i_c_a0, i_c_b0, i_c_c0 = abc_from_dq(float(id_c0), float(iq_c0))
    v_f_a0, v_f_b0, v_f_c0 = abc_from_dq(float(vd_f0), float(vq_f0))
    v_c_a0, v_c_b0, v_c_c0 = abc_from_dq(float(vd_c0), float(vq_c0))
    p_a0 = va0 * ia0
    p_b0 = vb0 * ib0
    p_c0 = vc0 * ic0
    q_a0 = (vb0 - vc0) * ia0
    q_b0 = (vc0 - va0) * ib0
    q_c0 = (va0 - vb0) * ic0
    p_conv0 = 0.5 * (vq_c0 * iq_c0 + vd_c0 * id_c0)
    d_va0 = 2.0 * np.pi * grid.fBase * (vc0 - vb0) / np.sqrt(3.0)
    d_vb0 = 2.0 * np.pi * grid.fBase * (va0 - vc0) / np.sqrt(3.0)
    d_vc0 = 2.0 * np.pi * grid.fBase * (vb0 - va0) / np.sqrt(3.0)

    for name, value in (
        ("theta", theta),
        ("omega", 1.0),
        ("P", p0),
        ("Q", q0),
        ("P_ref", p0),
        ("Q_ref", q0),
        ("y_p_lp", p0),
        ("y_q_lp", q0),
        ("V_ref", v_ref0),
        ("V", v_ref0),
        ("vd_ref", 0.0),
        ("vq_ref", v_ref0),
        ("vd_g", vd_g0),
        ("vq_g", vq_g0),
        ("vd_f", vd_f0),
        ("vq_f", vq_f0),
        ("vd_c", vd_c0),
        ("vq_c", vq_c0),
        ("vd_c_ref", vd_c0),
        ("vq_c_ref", vq_c0),
        ("id_g", id0),
        ("iq_g", iq0),
        ("id_c", id_c0),
        ("iq_c", iq_c0),
        ("id_ref", id_c0),
        ("iq_ref", iq_c0),
        ("id_ref_sat", id_c0),
        ("iq_ref_sat", iq_c0),
        ("z_vd_loop", z_vd0),
        ("z_vq_loop", z_vq0),
        ("z_id_loop", vd_ctrl0 / ki_icl),
        ("z_iq_loop", vq_ctrl0 / ki_icl),
        ("vd_ctrl_out", vd_ctrl0),
        ("vq_ctrl_out", vq_ctrl0),
        ("Pt_vsc", -p0),
        ("Qt_vsc", -q0),
        ("Pf_vsc", -p_conv0),
        ("Qf_vsc", 0.0),
        ("P_A", p_a0),
        ("P_B", p_b0),
        ("P_C", p_c0),
        ("Q_A", q_a0),
        ("Q_B", q_b0),
        ("Q_C", q_c0),
        ("d_v_A", d_va0),
        ("d_v_B", d_vb0),
        ("d_v_C", d_vc0),
        ("i_A", ia0),
        ("i_B", ib0),
        ("i_C", ic0),
        ("i_g_A", ia0),
        ("i_g_B", ib0),
        ("i_g_C", ic0),
        ("i_c_A", i_c_a0),
        ("i_c_B", i_c_b0),
        ("i_c_C", i_c_c0),
        ("v_f_A", v_f_a0),
        ("v_f_B", v_f_b0),
        ("v_f_C", v_f_c0),
        ("A_v_c", v_c_a0),
        ("B_v_c", v_c_b0),
        ("C_v_c", v_c_c0),
        (f"vd_{model_name}_bus", vd_g0),
        (f"vq_{model_name}_bus", vq_g0),
        (f"vd_{model_name}_ig", id0),
        (f"vq_{model_name}_ig", iq0),
        (f"vd_{model_name}_ic", id_c0),
        (f"vq_{model_name}_ic", iq_c0),
        (f"vd_{model_name}_vf", vd_f0),
        (f"vq_{model_name}_vf", vq_f0),
    ):
        set_runtime_value(problem, gfm_block, name, value)

    theta_var = find_name_in_block("theta", gfm_block)
    if theta_var is not None and theta_var.diff_var is not None:
        problem.diff_init_guess[theta_var.diff_var.uid] = 2.0 * np.pi * grid.fBase

    def dq_derivative_samples(d_value: float, q_value: float) -> tuple[float, float, float]:
        c = np.cos(theta)
        s = np.sin(theta)
        sqrt3 = np.sqrt(3.0)
        theta_dot = 2.0 * np.pi * grid.fBase * omega0
        return (
            theta_dot * (-d_value * s + q_value * c),
            theta_dot * (d_value * (0.5 * s + 0.5 * sqrt3 * c)
                         + q_value * (-0.5 * c + 0.5 * sqrt3 * s)),
            theta_dot * (d_value * (0.5 * s - 0.5 * sqrt3 * c)
                         + q_value * (-0.5 * c - 0.5 * sqrt3 * s)),
        )

    omega_base = 2.0 * np.pi * grid.fBase
    d_i_g_a0 = omega_base * (v_f_a0 - va0 - rc * ia0) / lc
    d_i_g_b0 = omega_base * (v_f_b0 - vb0 - rc * ib0) / lc
    d_i_g_c0 = omega_base * (v_f_c0 - vc0 - rc * ic0) / lc

    for base_name, value in (
        ("i_g_A", d_i_g_a0),
        ("i_g_B", d_i_g_b0),
        ("i_g_C", d_i_g_c0),
        ("i_c_A", dq_derivative_samples(float(id_c0), float(iq_c0))[0]),
        ("i_c_B", dq_derivative_samples(float(id_c0), float(iq_c0))[1]),
        ("i_c_C", dq_derivative_samples(float(id_c0), float(iq_c0))[2]),
        ("v_f_A", dq_derivative_samples(float(vd_f0), float(vq_f0))[0]),
        ("v_f_B", dq_derivative_samples(float(vd_f0), float(vq_f0))[1]),
        ("v_f_C", dq_derivative_samples(float(vd_f0), float(vq_f0))[2]),
        ("z_vd_loop", 0.0 - vd_f0),
        ("z_vq_loop", v_ref0 - vq_f0),
        ("z_id_loop", 0.0),
        ("z_iq_loop", 0.0),
    ):
        base_var = find_name_in_block(base_name, gfm_block)
        if base_var is not None and base_var.diff_var is not None:
            problem.diff_init_guess[base_var.diff_var.uid] = float(value)

    predictors = dict(getattr(problem, "algebraic_predictor_derivatives", {}))
    for name, value in (
        ("i_A", dq_derivative_samples(float(id0), float(iq0))[0]),
        ("i_B", dq_derivative_samples(float(id0), float(iq0))[1]),
        ("i_C", dq_derivative_samples(float(id0), float(iq0))[2]),
        ("A_v_c", dq_derivative_samples(float(vd_c0), float(vq_c0))[0]),
        ("B_v_c", dq_derivative_samples(float(vd_c0), float(vq_c0))[1]),
        ("C_v_c", dq_derivative_samples(float(vd_c0), float(vq_c0))[2]),
        ("vd_g", 0.0),
        ("vq_g", 0.0),
        ("vd_f", 0.0),
        ("vq_f", 0.0),
        ("vd_c", 0.0),
        ("vq_c", 0.0),
        ("id_g", 0.0),
        ("iq_g", 0.0),
        ("id_c", 0.0),
        ("iq_c", 0.0),
    ):
        var = find_name_in_block(name, gfm_block)
        if var is not None:
            predictors[var.uid] = float(value)
    problem.algebraic_predictor_derivatives = predictors


def seed_bus_algebraic_predictors(problem: EmtProblemDae, grid: vge.MultiCircuit, pf_results) -> None:
    predictors = dict(getattr(problem, "algebraic_predictor_derivatives", {}))
    omega_base = 2.0 * np.pi * grid.fBase
    for bus_idx, bus in enumerate(grid.buses):
        voltage = complex(pf_results.voltage[bus_idx])
        vpk = float(np.sqrt(2.0) * abs(voltage))
        angle = float(np.angle(voltage))
        va0 = vpk * np.sin(angle)
        vb0 = vpk * np.sin(angle - 2.0 * np.pi / 3.0)
        vc0 = vpk * np.sin(angle + 2.0 * np.pi / 3.0)
        derivative_by_name = {
            "v_A": omega_base * (vc0 - vb0) / np.sqrt(3.0),
            "v_B": omega_base * (va0 - vc0) / np.sqrt(3.0),
            "v_C": omega_base * (vb0 - va0) / np.sqrt(3.0),
        }
        for name, derivative in derivative_by_name.items():
            var = find_name_in_block(name, bus.emt_model)
            if var is not None:
                predictors[var.uid] = float(derivative)

    problem.algebraic_predictor_derivatives = predictors


def seed_line_current_predictors(problem: EmtProblemDae, grid: vge.MultiCircuit, pf_results, devices: dict) -> None:
    predictors = dict(getattr(problem, "algebraic_predictor_derivatives", {}))
    omega_base = 2.0 * np.pi * grid.fBase

    def bus_voltage_derivatives(bus) -> dict[str, float]:
        bus_idx = grid.buses.index(bus)
        voltage = complex(pf_results.voltage[bus_idx])
        vpk = float(np.sqrt(2.0) * abs(voltage))
        angle = float(np.angle(voltage))
        va0 = vpk * np.sin(angle)
        vb0 = vpk * np.sin(angle - 2.0 * np.pi / 3.0)
        vc0 = vpk * np.sin(angle + 2.0 * np.pi / 3.0)
        return {
            "A": omega_base * (vc0 - vb0) / np.sqrt(3.0),
            "B": omega_base * (va0 - vc0) / np.sqrt(3.0),
            "C": omega_base * (vb0 - va0) / np.sqrt(3.0),
        }

    def line_series_derivative(block, phase: str) -> float:
        for var in block.state_vars:
            if var.name.startswith("i_ser_") and var.name.endswith(f"_{phase}") and var.diff_var is not None:
                return float(problem.diff_init_guess.get(var.diff_var.uid, 0.0))
        return 0.0

    def set_line_current_derivative(block, prefix: str, phase: str, derivative: float) -> None:
        for var in block.algebraic_vars:
            if var.name.startswith(prefix) and var.name.endswith(f"_{phase}"):
                predictors[var.uid] = float(derivative)

    for line_key in ("line13", "line34"):
        if line_key not in devices:
            continue
        line = devices[line_key]
        block = line.emt_model
        d_v_from = bus_voltage_derivatives(line.bus_from)
        d_v_to = bus_voltage_derivatives(line.bus_to)
        for phase in ("A", "B", "C"):
            d_i_ser = line_series_derivative(block, phase)
            set_line_current_derivative(block, "if_", phase, d_i_ser + 1.0e-5 * d_v_from[phase])
            set_line_current_derivative(block, "it_", phase, -d_i_ser + 1.0e-5 * d_v_to[phase])

    problem.algebraic_predictor_derivatives = predictors


def seed_source_currents_from_equations(problem: EmtProblemDae, devices: dict) -> None:
    source = devices["gen_slack"].emt_model
    bus = devices["bus_slack"].emt_model

    def event_value(block, name: str, default: float = 0.0) -> float:
        var = find_name_in_block(name, block)
        uid2idx = getattr(problem, "_uid2idx_event_params", getattr(problem, "uid2idx_event_params", {}))
        if var is not None and var.uid in uid2idx:
            return float(problem._event_params_values[uid2idx[var.uid]])
        return float(default)

    def init_value(block, name: str) -> float:
        var = find_name_in_block(name, block)
        if var is None:
            return 0.0
        if var.uid in problem.init_guess:
            return float(problem.init_guess[var.uid])
        if var.uid in problem._uid2idx_vars:
            return float(problem.get_x0()[problem.get_var_idx(var)])
        return 0.0

    def set_init(block, name: str, value: float) -> None:
        var = find_name_in_block(name, block)
        if var is not None and var.uid in problem._uid2idx_vars:
            problem.init_guess[var.uid] = float(value)

    amplitude = event_value(source, "V_amp", 0.0)
    phase_a_deg = event_value(source, "phi_deg_A", 0.0)
    offset = event_value(source, "V_offset", 0.0)
    conductance = event_value(source, "g_src", 0.0)
    theta = init_value(source, "theta_src")
    shifts = {"A": 0.0, "B": -120.0, "C": 120.0}
    for phase, shift_deg in shifts.items():
        angle = theta + np.deg2rad(phase_a_deg + shift_deg)
        v_src = offset + amplitude * np.sin(angle)
        v_bus = init_value(bus, f"v_{phase}")
        set_init(source, f"i_{phase}", conductance * (v_src - v_bus))


def uid_bindings_from_problem(problem: EmtProblemDae) -> dict[int, float]:
    x0 = problem.get_x0()
    bindings = {}
    for var in list(problem.get_state_vars()) + list(problem.get_algebraic_vars()):
        if var.uid in problem._uid2idx_vars:
            bindings[var.uid] = float(x0[problem.get_var_idx(var)])

    uid2idx = getattr(problem, "_uid2idx_event_params", getattr(problem, "uid2idx_event_params", {}))
    for var in problem.get_variable_parameters():
        if var.uid in uid2idx:
            bindings[var.uid] = float(problem._event_params_values[uid2idx[var.uid]])
    return bindings


def print_gfm_algebraic_residuals(problem: EmtProblemDae, gfm_block, limit: int = 10) -> None:
    bindings = uid_bindings_from_problem(problem)

    gfm_var_names = {var.name for var in gfm_block.algebraic_vars + gfm_block.state_vars + gfm_block.diff_vars}
    gfm_var_names.update(var.name for var in gfm_block.in_vars + gfm_block.out_vars)
    gfm_var_names.update(var.name for var in gfm_block.event_dict.keys())
    for child in gfm_block.children:
        gfm_var_names.update(var.name for var in child.algebraic_vars + child.state_vars + child.diff_vars)
        gfm_var_names.update(var.name for var in child.in_vars + child.out_vars)
        gfm_var_names.update(var.name for var in child.event_dict.keys())

    residuals = []
    for eq in problem.get_algebraic_eqs():
        eq_text = str(eq)
        if not any(name in eq_text for name in gfm_var_names):
            continue
        try:
            value = float(eq.eval_uid(bindings))
        except Exception:
            continue
        if np.isfinite(value):
            residuals.append((abs(value), value, eq_text[:160]))

    residuals.sort(reverse=True, key=lambda item: item[0])
    max_res = residuals[0][0] if residuals else 0.0
    print(f"GFM algebraic residual at seeded x0: max_abs={max_res:.6e}")
    for abs_value, value, label in residuals[:limit]:
        print(f"  residual={value:.6e}, abs={abs_value:.6e}, eq={label}")


def print_gfm_state_residuals(problem: EmtProblemDae, gfm_block, limit: int = 16) -> None:
    bindings = uid_bindings_from_problem(problem)

    residuals = []
    for state_var, diff_var, eq in zip(gfm_block.state_vars, gfm_block.diff_vars, gfm_block.state_eqs):
        try:
            rhs = float(eq.eval_uid(bindings))
        except Exception:
            continue
        diff_value = float(problem.diff_init_guess.get(diff_var.uid, 0.0))
        residual = rhs - diff_value
        residuals.append((abs(residual), residual, rhs, diff_value, state_var.name))

    residuals.sort(reverse=True, key=lambda item: item[0])
    max_res = residuals[0][0] if residuals else 0.0
    print(f"GFM state residual at seeded x0: max_abs={max_res:.6e}")
    for abs_value, residual, rhs, diff_value, name in residuals[:limit]:
        print(
            f"  {name}: rhs={rhs:.6e}, d_init={diff_value:.6e}, "
            f"residual={residual:.6e}, abs={abs_value:.6e}"
        )


def print_block_algebraic_residuals(problem: EmtProblemDae, block, label: str, limit: int = 10) -> None:
    bindings = uid_bindings_from_problem(problem)
    block_var_names = {var.name for var in block.algebraic_vars + block.state_vars + block.diff_vars}
    block_var_names.update(var.name for var in block.in_vars + block.out_vars)
    block_var_names.update(var.name for var in block.event_dict.keys())
    for child in block.children:
        block_var_names.update(var.name for var in child.algebraic_vars + child.state_vars + child.diff_vars)
        block_var_names.update(var.name for var in child.in_vars + child.out_vars)
        block_var_names.update(var.name for var in child.event_dict.keys())

    residuals = []
    for eq in problem.get_algebraic_eqs():
        eq_text = str(eq)
        if not any(name in eq_text for name in block_var_names):
            continue
        try:
            value = float(eq.eval_uid(bindings))
        except Exception:
            continue
        if np.isfinite(value):
            residuals.append((abs(value), value, eq_text[:160]))

    residuals.sort(reverse=True, key=lambda item: item[0])
    max_res = residuals[0][0] if residuals else 0.0
    print(f"{label} algebraic residual at seeded x0: max_abs={max_res:.6e}")
    for abs_value, value, eq_text in residuals[:limit]:
        print(f"  residual={value:.6e}, abs={abs_value:.6e}, eq={eq_text}")


def print_block_state_residuals(problem: EmtProblemDae, block, label: str, limit: int = 10) -> None:
    bindings = uid_bindings_from_problem(problem)
    residuals = []
    for state_var, diff_var, eq in zip(block.state_vars, block.diff_vars, block.state_eqs):
        try:
            rhs = float(eq.eval_uid(bindings))
        except Exception:
            continue
        diff_value = float(problem.diff_init_guess.get(diff_var.uid, 0.0))
        residual = rhs - diff_value
        residuals.append((abs(residual), residual, rhs, diff_value, state_var.name))

    residuals.sort(reverse=True, key=lambda item: item[0])
    max_res = residuals[0][0] if residuals else 0.0
    print(f"{label} state residual at seeded x0: max_abs={max_res:.6e}")
    for abs_value, residual, rhs, diff_value, name in residuals[:limit]:
        print(
            f"  {name}: rhs={rhs:.6e}, d_init={diff_value:.6e}, "
            f"residual={residual:.6e}, abs={abs_value:.6e}"
        )


def print_global_state_residuals(problem: EmtProblemDae, limit: int = 20) -> None:
    bindings = uid_bindings_from_problem(problem)

    residuals = []
    for state_var, diff_var, eq in zip(problem.get_state_vars(), problem.get_diff_vars(), problem.get_state_eqs()):
        try:
            rhs = float(eq.eval_uid(bindings))
        except Exception:
            continue
        diff_value = float(problem.diff_init_guess.get(diff_var.uid, 0.0))
        residual = rhs - diff_value
        residuals.append((abs(residual), residual, rhs, diff_value, state_var.name))

    residuals.sort(reverse=True, key=lambda item: item[0])
    max_res = residuals[0][0] if residuals else 0.0
    print(f"Global state residual at seeded x0: max_abs={max_res:.6e}")
    for abs_value, residual, rhs, diff_value, name in residuals[:limit]:
        print(
            f"  {name}: rhs={rhs:.6e}, d_init={diff_value:.6e}, "
            f"residual={residual:.6e}, abs={abs_value:.6e}"
        )


def print_bus_kcl_residuals(problem: EmtProblemDae, grid: vge.MultiCircuit) -> None:
    uid_bindings = uid_bindings_from_problem(problem)

    phases = ("N", "A", "B", "C")
    print("Bus KCL residuals at seeded x0:")
    working_models = getattr(problem, "_working_emt_models", {})
    for bus in grid.buses:
        bus_model = working_models.get(str(bus.idtag), bus.emt_model)
        residuals = []
        for eq in bus_model.algebraic_eqs:
            try:
                residuals.append(float(eq.eval_uid(uid_bindings)))
            except Exception:
                residuals.append(float("nan"))

        phase_vars = [
            find_name_in_block("v_N", bus_model),
            find_name_in_block("v_A", bus_model),
            find_name_in_block("v_B", bus_model),
            find_name_in_block("v_C", bus_model),
        ]
        active_phases = [phase for phase, var in zip(phases, phase_vars) if var is not None]
        labels = active_phases if len(active_phases) == len(residuals) else [f"eq{idx}" for idx in range(len(residuals))]
        formatted = ", ".join(
            f"{label}={value:.6e}" for label, value in zip(labels, residuals)
        )
        max_abs = max((abs(value) for value in residuals if np.isfinite(value)), default=0.0)
        print(f"  {bus.name}: max_abs={max_abs:.6e}; {formatted}")


def print_case_kcl_residuals(problem: EmtProblemDae, devices: dict, gfm_block) -> None:
    x0 = problem.get_x0()

    print_case_kcl_from_vector(problem, devices, gfm_block, x0, "seeded x0")


def print_case_kcl_from_vector(problem: EmtProblemDae, devices: dict, gfm_block, vector: np.ndarray, label: str) -> None:
    vector = np.asarray(vector, dtype=float)
    working_models = getattr(problem, "_working_emt_models", {})

    def model(device):
        return working_models.get(str(device.idtag), device.emt_model)

    def value(block, name: str) -> float:
        var = find_name_in_block(name, block)
        if var is None or var.uid not in problem._uid2idx_vars:
            return 0.0
        return float(vector[problem.get_var_idx(var)])

    print(f"Case KCL residuals from device currents at {label}:")
    for phase in ("A", "B", "C"):
        line13_if = value(model(devices["line13"]), f"if_Line13_{phase}")
        line13_it = value(model(devices["line13"]), f"it_Line13_{phase}")
        line34_if = value(model(devices["line34"]), f"if_Line_3-4_{phase}") if "line34" in devices else 0.0
        line34_it = value(model(devices["line34"]), f"it_Line_3-4_{phase}") if "line34" in devices else 0.0
        slack_i = value(model(devices["gen_slack"]), f"i_{phase}")
        gfm_i = value(gfm_block, f"i_{phase}")
        load_i = value(model(devices["load"]), f"i_{phase}") if "load" in devices else 0.0

        slack_bus = -line13_it + slack_i
        gfm_bus = -line13_if - line34_if + gfm_i
        load_bus = -line34_it + load_i
        if "load" in devices:
            print(
                f"  phase {phase}: "
                f"slack={slack_bus:.6e}, gfm={gfm_bus:.6e}, load={load_bus:.6e}"
            )
        else:
            print(f"  phase {phase}: slack={slack_bus:.6e}, gfm={gfm_bus:.6e}")


def print_gfm_bus_kcl_derivative(problem: EmtProblemDae, devices: dict, gfm_block) -> None:
    working_models = getattr(problem, "_working_emt_models", {})

    def model(device):
        return working_models.get(str(device.idtag), device.emt_model)

    def derivative_of_state(block, state_prefix: str, phase: str) -> float:
        for var in block.state_vars:
            if var.name.startswith(state_prefix) and var.name.endswith(f"_{phase}") and var.diff_var is not None:
                return float(problem.diff_init_guess.get(var.diff_var.uid, 0.0))
        return 0.0

    def derivative_of_gfm_current(phase: str) -> float:
        var = find_name_in_block(f"i_g_{phase}", gfm_block)
        if var is not None and var.diff_var is not None:
            return float(problem.diff_init_guess.get(var.diff_var.uid, 0.0))
        return 0.0

    print("GFM bus KCL derivative residual at seeded x0:")
    for phase in ("A", "B", "C"):
        d_if_13 = derivative_of_state(model(devices["line13"]), "i_ser_", phase)
        d_if_34 = derivative_of_state(model(devices["line34"]), "i_ser_", phase) if "line34" in devices else 0.0
        d_i_gfm = derivative_of_gfm_current(phase)
        residual = -d_if_13 - d_if_34 + d_i_gfm
        print(
            f"  phase {phase}: -d_if13={-d_if_13:.6e}, -d_if34={-d_if_34:.6e}, "
            f"d_i_gfm={d_i_gfm:.6e}, residual={residual:.6e}"
        )


def trace(problem: EmtProblemDae, y: np.ndarray, var_name: str, block) -> np.ndarray:
    var = find_name_in_block(var_name, block)
    if var is None or var.uid not in problem._uid2idx_vars:
        raise RuntimeError(f"Variable {var_name} is not in the solved EMT vector")
    return np.asarray(y[:, problem.get_var_idx(var)], dtype=float)


def trace_or_param(problem: EmtProblemDae, y: np.ndarray, var_name: str, block) -> np.ndarray:
    var = find_name_in_block(var_name, block)
    if var is None:
        raise RuntimeError(f"Variable {var_name} is not in the EMT model")
    if var.uid in problem._uid2idx_vars:
        return np.asarray(y[:, problem.get_var_idx(var)], dtype=float)

    uid2idx = getattr(problem, "_uid2idx_event_params", getattr(problem, "uid2idx_event_params", {}))
    if var.uid in uid2idx:
        value = float(problem._event_params_values[uid2idx[var.uid]])
        return np.full(y.shape[0], value, dtype=float)

    raise RuntimeError(f"Variable {var_name} is not in the solved EMT vector or event parameters")


def save_plot(filename: str) -> None:
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, filename), dpi=150)
    plt.close()


def plot_results(t: np.ndarray, problem: EmtProblemDae, y: np.ndarray, gfm_block, bus_gfm) -> None:
    t = np.asarray(t, dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(t, trace(problem, y, "P", gfm_block), label="P")
    plt.plot(t, trace(problem, y, "Q", gfm_block), label="Q")
    plt.xlabel("Time (s)")
    plt.ylabel("Power (pu)")
    plt.title("EMT GFM Power")
    plt.xlim(float(t[0]), float(t[-1]))
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    save_plot("gfm_power.png")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, trace(problem, y, "omega", gfm_block), label="omega")
    axes[0].set_ylabel("Frequency (pu)")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, trace(problem, y, "V", gfm_block), label="V")
    axes[1].set_ylabel("Voltage ref (pu)")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, trace(problem, y, "v_A", bus_gfm.emt_model), label="v_A")
    axes[2].plot(t, trace(problem, y, "v_B", bus_gfm.emt_model), label="v_B")
    axes[2].plot(t, trace(problem, y, "v_C", bus_gfm.emt_model), label="v_C")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Voltage (pu peak)")
    axes[2].grid(True)
    axes[2].legend(loc="upper right")
    axes[2].set_xlim(float(t[0]), float(t[-1]))

    fig.suptitle("EMT GFM States and PCC Voltages")
    fig.tight_layout()
    save_plot("gfm_states_and_voltages.png")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, trace(problem, y, "vd_g", gfm_block), label="vd_g")
    axes[0].plot(t, trace(problem, y, "vq_g", gfm_block), label="vq_g")
    axes[0].set_ylabel("Grid dq voltage")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, trace(problem, y, "vd_f", gfm_block), label="vd_f")
    axes[1].plot(t, trace(problem, y, "vq_f", gfm_block), label="vq_f")
    axes[1].plot(t, trace(problem, y, "vd_ref", gfm_block), "--", label="vd_ref")
    axes[1].plot(t, trace(problem, y, "vq_ref", gfm_block), "--", label="vq_ref")
    axes[1].set_ylabel("Filter dq voltage")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, trace(problem, y, "vd_c", gfm_block), label="vd_c")
    axes[2].plot(t, trace(problem, y, "vq_c", gfm_block), label="vq_c")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Converter dq voltage")
    axes[2].grid(True)
    axes[2].legend(loc="upper right")
    axes[2].set_xlim(float(t[0]), float(t[-1]))
    fig.suptitle("EMT GFM dq Voltages")
    fig.tight_layout()
    save_plot("gfm_dq_voltages.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, trace(problem, y, "id_g", gfm_block), label="id_g")
    axes[0].plot(t, trace(problem, y, "iq_g", gfm_block), label="iq_g")
    axes[0].set_ylabel("Grid dq current")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, trace(problem, y, "id_c", gfm_block), label="id_c")
    axes[1].plot(t, trace(problem, y, "iq_c", gfm_block), label="iq_c")
    axes[1].plot(t, trace(problem, y, "id_ref", gfm_block), "--", label="id_ref")
    axes[1].plot(t, trace(problem, y, "iq_ref", gfm_block), "--", label="iq_ref")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Converter dq current")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(float(t[0]), float(t[-1]))
    fig.suptitle("EMT GFM dq Currents")
    fig.tight_layout()
    save_plot("gfm_dq_currents.png")

    cf = trace_or_param(problem, y, "Cf", gfm_block)
    rcap = trace_or_param(problem, y, "Rcap", gfm_block)
    omega = trace(problem, y, "omega", gfm_block)
    id_g = trace(problem, y, "id_g", gfm_block)
    iq_g = trace(problem, y, "iq_g", gfm_block)
    id_c = trace(problem, y, "id_c", gfm_block)
    iq_c = trace(problem, y, "iq_c", gfm_block)
    vd_f = trace(problem, y, "vd_f", gfm_block)
    vq_f = trace(problem, y, "vq_f", gfm_block)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, id_c - id_g, label="id_c - id_g")
    axes[0].plot(t, -cf * omega * vq_f + vd_f / rcap, "--", label="-Cf*omega*vq_f + vd_f/Rcap")
    axes[0].plot(t, id_c - id_g + cf * omega * vq_f - vd_f / rcap, ":", label="d-axis residual")
    axes[0].set_ylabel("d capacitor current")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, iq_c - iq_g, label="iq_c - iq_g")
    axes[1].plot(t, cf * omega * vd_f + vq_f / rcap, "--", label="Cf*omega*vd_f + vq_f/Rcap")
    axes[1].plot(t, iq_c - iq_g - cf * omega * vd_f - vq_f / rcap, ":", label="q-axis residual")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("q capacitor current")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(float(t[0]), float(t[-1]))
    fig.suptitle("EMT GFM LCL Capacitor Current Balance")
    fig.tight_layout()
    save_plot("gfm_lcl_capacitor_current_balance.png")

    p = trace(problem, y, "P", gfm_block)
    q = trace(problem, y, "Q", gfm_block)
    p_lp = trace(problem, y, "y_p_lp", gfm_block)
    q_lp = trace(problem, y, "y_q_lp", gfm_block)
    p_ref = trace_or_param(problem, y, "P_ref", gfm_block)
    if P_REF_STEP is not None:
        # Runtime parameters retain only their final value after a segmented
        # event run. Reconstruct the pre-event reference for truthful plots.
        p_ref = p_ref.copy()
        p_ref[t < P_REF_STEP_TIME] -= float(P_REF_STEP)
    q_ref = trace_or_param(problem, y, "Q_ref", gfm_block)
    v = trace(problem, y, "V", gfm_block)
    v_ref = trace_or_param(problem, y, "V_ref", gfm_block)
    vd_ref = trace(problem, y, "vd_ref", gfm_block)
    vq_ref = trace(problem, y, "vq_ref", gfm_block)
    vd_f = trace(problem, y, "vd_f", gfm_block)
    vq_f = trace(problem, y, "vq_f", gfm_block)
    id_ref = trace(problem, y, "id_ref", gfm_block)
    iq_ref = trace(problem, y, "iq_ref", gfm_block)
    id_c = trace(problem, y, "id_c", gfm_block)
    iq_c = trace(problem, y, "iq_c", gfm_block)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(t, p, label="P")
    axes[0].plot(t, p_lp, label="P_lowpass")
    axes[0].plot(t, p_ref, "--", label="P_ref")
    axes[0].set_ylabel("P loop input")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, q, label="Q")
    axes[1].plot(t, q_lp, label="Q_lowpass")
    axes[1].plot(t, q_ref, "--", label="Q_ref")
    axes[1].set_ylabel("Q loop input")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, p_lp - p_ref, label="P_lowpass - P_ref")
    axes[2].plot(t, q_lp - q_ref, label="Q_lowpass - Q_ref")
    axes[2].plot(t, v - v_ref, label="V - V_ref")
    axes[2].set_ylabel("Outer errors")
    axes[2].grid(True)
    axes[2].legend(loc="upper right")

    axes[3].plot(t, vd_ref - vd_f, label="vd_ref - vd_f")
    axes[3].plot(t, vq_ref - vq_f, label="vq_ref - vq_f")
    axes[3].plot(t, id_ref - id_c, label="id_ref - id_c")
    axes[3].plot(t, iq_ref - iq_c, label="iq_ref - iq_c")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Inner errors")
    axes[3].grid(True)
    axes[3].legend(loc="upper right")
    axes[3].set_xlim(float(t[0]), float(t[-1]))
    fig.suptitle("EMT GFM Controller Inputs and Errors")
    fig.tight_layout()
    save_plot("gfm_controller_inputs_errors.png")

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(t, trace(problem, y, "z_vd_loop", gfm_block), label="z_vd_loop")
    axes[0].plot(t, trace(problem, y, "z_vq_loop", gfm_block), label="z_vq_loop")
    axes[0].set_ylabel("Voltage PI states")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, trace(problem, y, "z_id_loop", gfm_block), label="z_id_loop")
    axes[1].plot(t, trace(problem, y, "z_iq_loop", gfm_block), label="z_iq_loop")
    axes[1].set_ylabel("Current PI states")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, trace(problem, y, "vd_ctrl_out", gfm_block), label="vd_ctrl_out")
    axes[2].plot(t, trace(problem, y, "vq_ctrl_out", gfm_block), label="vq_ctrl_out")
    axes[2].set_ylabel("Current ctrl output")
    axes[2].grid(True)
    axes[2].legend(loc="upper right")

    axes[3].plot(t, trace(problem, y, "id_ref", gfm_block), label="id_ref")
    axes[3].plot(t, trace(problem, y, "iq_ref", gfm_block), label="iq_ref")
    axes[3].plot(t, trace(problem, y, "id_c", gfm_block), "--", label="id_c")
    axes[3].plot(t, trace(problem, y, "iq_c", gfm_block), "--", label="iq_c")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Current refs/states")
    axes[3].grid(True)
    axes[3].legend(loc="upper right")
    axes[3].set_xlim(float(t[0]), float(t[-1]))
    fig.suptitle("EMT GFM Control Loop States")
    fig.tight_layout()
    save_plot("gfm_control_loop_states.png")


def main() -> None:
    grid, devices = create_simplified_grid()
    if "load" in devices and LOAD_SCALE != 1.0:
        devices["load"].P *= LOAD_SCALE
        devices["load"].Q *= LOAD_SCALE
        print(f"Scaled load powers by {LOAD_SCALE:g}")

    pf_results = run_power_flow(grid)
    print(f"Power flow converged: {pf_results.converged}")
    print(pf_results.get_bus_df())
    print(pf_results.get_branch_df())

    gfm_block = attach_emt_models(grid, devices, pf_results=pf_results)
    apply_gfm_gain_overrides(gfm_block, grid.var_factory)

    options = EmtOptions(
        time_step=TIME_STEP,
        simulation_time=SIM_TIME,
        tolerance=1.0e-6,
        integration_method=integration_method_from_env(),
        solver_type=EmtSolverTypes.StructuralAD,
        initialization_method=EmtInitializationMethod.Explicit,
        init_newton_max_iter=int(os.environ.get("VERAGRID_GFM_EMT_NEWTON_MAX_ITER", "200")),
        verbose=0,
    )
    problem = EmtProblemDae(grid=grid, options=options, pf_results=pf_results)
    print_gfm_algebraic_residuals(problem, gfm_block)
    print_gfm_state_residuals(problem, gfm_block)
    print_block_algebraic_residuals(problem, devices["gen_slack"].emt_model, "Slack source")
    print_block_state_residuals(problem, devices["gen_slack"].emt_model, "Slack source")
    print_global_state_residuals(problem)
    print_bus_kcl_residuals(problem, grid)
    print_case_kcl_residuals(problem, devices, gfm_block)
    print_gfm_bus_kcl_derivative(problem, devices, gfm_block)

    try:
        if WARMUP_TIME > 0.0:
            print(f"Running EMT warm-up: {WARMUP_TIME:.6g} s")
            tw, yw, dyw, warm_well_initialized, warm_converged = run_emt(problem, options, WARMUP_TIME)
            print(f"EMT warm-up well initialized: {warm_well_initialized}")
            print(f"EMT warm-up converged: {warm_converged}")
            if warm_well_initialized and warm_converged:
                apply_solution_as_initial_guess(problem, yw, dyw)
            else:
                print("Skipping warm-up state reuse because warm-up did not converge cleanly")
        if P_REF_STEP is not None:
            t, y, dy, well_initialized, converged = run_emt_with_p_ref_step(
                problem=problem,
                options=options,
                gfm_block=gfm_block,
                step_time=P_REF_STEP_TIME,
                step_delta=float(P_REF_STEP),
            )
        else:
            print("Running EMT simulation")
            t, y, dy, well_initialized, converged = run_emt(problem, options, options.simulation_time)
    except np.linalg.LinAlgError as exc:
        print(f"EMT solve failed during Newton iteration: {exc}")
        return

    print(f"EMT well initialized: {well_initialized}")
    print(f"EMT converged: {converged}")
    print(f"Solved samples: {len(t)}")
    print_case_kcl_from_vector(problem, devices, gfm_block, y[0, :], "solution t0")
    print_case_kcl_from_vector(problem, devices, gfm_block, y[-1, :], "solution final")

    for name in ("P", "Q", "omega", "V"):
        values = trace(problem, y, name, gfm_block)
        print(f"{name}: t0={values[0]: .9e}, final={values[-1]: .9e}")

    for name in (
        "i_A", "i_B", "i_C", "i_g_A", "i_g_B", "i_g_C",
        "vd_g", "vq_g", "id_g", "iq_g", "id_c", "iq_c", "id_ref", "iq_ref", "vd_c", "vq_c",
    ):
        values = trace(problem, y, name, gfm_block)
        print(f"{name}: t0={values[0]: .9e}, final={values[-1]: .9e}")
    for label, lhs_name, rhs_name in (
        ("vd_ref-vd_f", "vd_ref", "vd_f"),
        ("vq_ref-vq_f", "vq_ref", "vq_f"),
        ("id_ref-id_c", "id_ref", "id_c"),
        ("iq_ref-iq_c", "iq_ref", "iq_c"),
    ):
        lhs = trace(problem, y, lhs_name, gfm_block)
        rhs = trace(problem, y, rhs_name, gfm_block)
        diff = lhs - rhs
        print(f"{label}: t0={diff[0]: .9e}, final={diff[-1]: .9e}")
    for name in ("z_vd_loop", "z_vq_loop", "z_id_loop", "z_iq_loop", "vd_ctrl_out", "vq_ctrl_out"):
        values = trace(problem, y, name, gfm_block)
        print(f"{name}: t0={values[0]: .9e}, final={values[-1]: .9e}")
    for name in ("v_A", "v_B", "v_C"):
        values = trace(problem, y, name, devices["bus_gfm"].emt_model)
        print(f"bus {name}: t0={values[0]: .9e}, final={values[-1]: .9e}")
    for name in ("i_A", "i_B", "i_C", "e_A", "e_B", "e_C", "Pe", "Qe"):
        var = find_name_in_block(name, devices["gen_slack"].emt_model)
        if var is None or var.uid not in problem._uid2idx_vars:
            continue
        values = trace(problem, y, name, devices["gen_slack"].emt_model)
        print(f"slack {name}: t0={values[0]: .9e}, final={values[-1]: .9e}")

    plot_results(t, problem, y, gfm_block, devices["bus_gfm"])
    print(f"Plots saved to: {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
