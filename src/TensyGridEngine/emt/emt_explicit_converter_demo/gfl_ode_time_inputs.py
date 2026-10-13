"""Experimental standalone simulation helpers for the algebraic-free GFL ODE.

This module is deliberately separate from ``single_model_time_inputs.py``,
which exercises the established block-based GFM/GFL EMT models.
"""

from __future__ import annotations

import numpy as np
from typing import Mapping

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.problems.emt_model_problem import EmtModelProblem
from VeraGridEngine.Templates.Emt.vsc_gfl_ode import build_gfl_converter_model_ode
from VeraGridEngine.Utils.Symbolic.symbolic import Expr, Var
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtSolverTypes,
)

def _park_initial(values, angle: float):
    a, b, c = values
    cosine, sine = np.cos(angle), np.sin(angle)
    return (
        (2.0 * cosine * a + (-cosine + np.sqrt(3.0) * sine) * b
         + (-cosine - np.sqrt(3.0) * sine) * c) / 3.0,
        (2.0 * sine * a + (-sine - np.sqrt(3.0) * cosine) * b
         + (-sine + np.sqrt(3.0) * cosine) * c) / 3.0,
    )


def build_gfl_ode_problem(*, multilinear: bool = True, p0: float = 0.0,
                          q0: float = 0.0, voltage: complex = 1.0 + 0.0j,
                          vdc0: float = 2.0,
                          prescribed_inputs: Mapping[str, Expr],
                          time_variable: Var,
                          dc_state: str = "energy",
                          enable_current_limiter: bool = False,
                          simulation_time: float = 5.0e-2) -> EmtModelProblem:
    """Build the experimental algebraic-free GFL at a supplied operating point.

    ``prescribed_inputs`` maps each model input name to a symbolic expression.
    ``time_variable`` must be the same Var used in those expressions.
    """
    vf = VarFactory()
    time = time_variable
    block, _observables = build_gfl_converter_model_ode(
        vf, multilinear=multilinear,
        enable_current_limiter=enable_current_limiter,
        dc_state=dc_state,
        fixed_dc_voltage=vdc0,
    )

    voltage_angle = float(np.angle(voltage))
    voltage_peak = float(np.sqrt(2.0) * abs(voltage))
    i_q0 = 2.0 * p0 / voltage_peak
    i_d0 = -2.0 * q0 / voltage_peak
    phase_angles = (
        voltage_angle,
        voltage_angle - 2.0 * np.pi / 3.0,
        voltage_angle + 2.0 * np.pi / 3.0,
    )
    current_initial = [
        i_q0 * np.sin(phase_angle) + i_d0 * np.cos(phase_angle)
        for phase_angle in phase_angles
    ]
    current_derivative_initial = [
        2.0 * np.pi * 50.0
        * (i_q0 * np.cos(phase_angle) - i_d0 * np.sin(phase_angle))
        for phase_angle in phase_angles
    ]
    grid_voltage_initial = [voltage_peak * np.sin(phase_angle) for phase_angle in phase_angles]
    converter_voltage_initial = [
        vg + 0.01 * current + (0.10 / (2.0 * np.pi * 50.0)) * derivative
        for vg, current, derivative in zip(
            grid_voltage_initial, current_initial, current_derivative_initial
        )
    ]
    vg_d0, vg_q0 = _park_initial(grid_voltage_initial, voltage_angle)
    vc_d0, vc_q0 = _park_initial(converter_voltage_initial, voltage_angle)
    initial_values = {
        "i_line_A": current_initial[0],
        "i_line_B": current_initial[1],
        "i_line_C": current_initial[2],
        "xi_PLL": 0.0,
        "xi_Pac_ctrl": i_q0,
        "xi_Qac_ctrl": i_d0,
        "xi_vd_hat": vc_d0 - vg_d0 - 0.10 * i_q0,
        "xi_vq_hat": vc_q0 - vg_q0 + 0.10 * i_d0,
    }
    if dc_state != "stiff_voltage":
        initial_values["E_dc" if dc_state == "energy" else "Vdc_"] = (
            0.5 * 10.0 * vdc0 ** 2 if dc_state == "energy" else vdc0
        )
    if multilinear:
        for role in ("v", "i", "o"):
            initial_values[f"u_cos_{role}"] = np.cos(voltage_angle)
            initial_values[f"u_sin_{role}"] = np.sin(voltage_angle)
        if enable_current_limiter:
            current_limit = 2.5
            magnitude = float(np.hypot(i_d0, i_q0))
            scale = min(1.0, (1.0 - 1.0e-9) * current_limit / max(magnitude, 1.0e-12))
            i_d_limited0 = scale * i_d0
            i_q_limited0 = scale * i_q0
            for role in ("out", "a", "b"):
                initial_values[f"i_d_ref_sat_{role}"] = i_d_limited0
                initial_values[f"i_q_ref_sat_{role}"] = i_q_limited0
    else:
        initial_values["theta"] = voltage_angle

    expected = {variable.name for variable in block.in_vars}
    actual = set(prescribed_inputs)
    if actual != expected:
        raise ValueError(f"prescribed_inputs must contain exactly {sorted(expected)}; got {sorted(actual)}")
    input_expressions = {
        variable: prescribed_inputs[variable.name] for variable in block.in_vars
    }

    options = EmtOptions(
        time_step=1.0e-5,
        simulation_time=simulation_time,
        solver_type=EmtSolverTypes.Automatic,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
    )
    return EmtModelProblem(
        block,
        inputs=input_expressions,
        initial_values=initial_values,
        options=options,
        glob_time=time,
    )


def dc_voltage_trace(problem: EmtModelProblem, result, capacitance: float = 10.0):
    """Return Vdc for either supported DC state coordinate."""
    state_names = {variable.name for variable in problem.get_state_vars()}
    if "Vdc_" in state_names:
        return problem.trace(result, "Vdc_")
    energy = problem.trace(result, "E_dc")
    return np.sqrt(2.0 * energy / capacitance)


def runtime_input_traces(problem: EmtModelProblem, time: np.ndarray,
                         seed: int | None = None) -> dict[str, np.ndarray]:
    """Replay symbolic inputs; pass the simulation seed for random inputs."""
    if seed is not None:
        np.random.seed(seed)
    variables = list(problem.sys_block.in_vars)
    traces = {variable.name: np.zeros(len(time)) for variable in variables}
    values = problem.event_params_values.copy()
    for sample_index, sample in enumerate(time):
        values = problem.def_event_params_fn(values, float(sample))
        for variable in variables:
            traces[variable.name][sample_index] = values[
                problem.uid2idx_event_params[variable.uid]
            ]
    return traces
