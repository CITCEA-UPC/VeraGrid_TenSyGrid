"""Minimal standalone EMT-model simulations with time-dependent inputs."""

from __future__ import annotations

from pathlib import Path
import sys

# Allow this demo to run directly by path without installing the source tree.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.problems.emt_model_problem import EmtModelProblem
from VeraGridEngine.Templates.Emt.emt_gfm_upc import build_emt_gfm_aggregated_model
from VeraGridEngine.Templates.Emt.vsc_gfl_emt import build_gfl_converter_model_emt
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const, heaviside, rand, sin
from VeraGridEngine.enumerations import ConverterControlType, DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes

OUTPUT = Path(__file__).with_name("single_model_time_inputs.png")
NOISE_SEED = 20260914
AC_VOLTAGE_NOISE_RMS_PU = 0.01
NOISE_START = 0.02


def balanced_abc(time, peak: float = np.sqrt(2.0), frequency: float = 50.0,
                 angle: float = 0.0, noise_rms: float = AC_VOLTAGE_NOISE_RMS_PU,
                 noise_kind: str = "white"):
    """Return grid phase expressions with selectable symbolic disturbances.

    ``white`` adds independent native ``rand(time)`` noise to each phase.
    ``grid_colored`` applies common multi-tone magnitude and phase modulation,
    preserving a balanced positive-sequence voltage.  Its frequencies bracket
    the outer- and inner-controller response range instead of injecting mostly
    solver-Nyquist energy that the filter naturally rejects.
    """
    from VeraGridEngine.Utils.Symbolic.symbolic import rand

    angle_expr = Const(2.0 * np.pi * frequency) * time + Const(angle)
    shift = Const(2.0 * np.pi / 3.0)
    phase_angles = angle_expr, angle_expr - shift, angle_expr + shift
    clean = tuple(peak * sin(phase) for phase in phase_angles)
    if noise_rms <= 0.0:
        return clean
    noise_gate = heaviside(time - Const(NOISE_START))
    if noise_kind == "grid_colored":
        frequencies = (5.0, 15.0, 35.0)
        magnitude_phases = (0.31, 2.07, 4.19)
        angle_phases = (1.13, 3.41, 5.27)
        rms_normalization = Const(np.sqrt(2.0 / len(frequencies)))
        magnitude_noise = Const(0.0)
        angle_noise = Const(0.0)
        for noise_frequency, magnitude_phase, angle_phase in zip(
                frequencies, magnitude_phases, angle_phases):
            noise_angle = Const(2.0 * np.pi * noise_frequency) * time
            magnitude_noise += sin(noise_angle + Const(magnitude_phase))
            angle_noise += sin(noise_angle + Const(angle_phase))
        magnitude_noise *= Const(noise_rms) * rms_normalization
        # A smaller common phase modulation accompanies the magnitude noise.
        angle_noise *= Const(0.5 * noise_rms) * rms_normalization
        return tuple(
            peak * (Const(1.0) + noise_gate * magnitude_noise)
            * sin(phase + noise_gate * angle_noise)
            for phase in phase_angles
        )
    if noise_kind != "white":
        raise ValueError("noise_kind must be 'white' or 'grid_colored'")
    return tuple(
        phase + noise_gate * Const(np.sqrt(12.0) * noise_rms) * (rand(time) - Const(0.5))
        for phase in clean
    )


def build_problem(kind: str, multilinear: bool = True, *, p0: float = 0.0,
                  q0: float = 0.0, voltage: complex = 1.0 + 0.0j,
                  vdc0: float = 2.0, voltage_noise_rms: float = AC_VOLTAGE_NOISE_RMS_PU,
                  voltage_noise_kind: str = "white",
                  p_noise_rms: float = 0.0, q_noise_rms: float = 0.0,
                  simulation_time: float = 5.0e-2,
                  dc_side: str = "power_balance") -> EmtModelProblem:
    """Build one externally driven GFM or GFL model problem."""
    vf = VarFactory()
    time = vf.add_var("single_model_time")
    initial_values = {}
    if kind.lower() == "gfm":
        block = build_emt_gfm_aggregated_model(vf, name="standalone_gfm", multilinear=multilinear)
    elif kind.lower() == "gfl":
        gfl_inputs = [vf.add_var(name) for name in (
            "vc_A", "vc_B", "vc_C", "vg_A", "vg_B", "vg_C",
            "i_line_A", "i_line_B", "i_line_C", "Vdc_",
            "Pt_vsc", "Qt_vsc", "Vpk_ref_input", "phi_v_ref_input",
        )]
        controller, p_measured, _q_measured = build_gfl_converter_model_emt(
            vf,
            inputs=gfl_inputs,
            control1=ConverterControlType.Pac,
            control2=ConverterControlType.Qac,
            multilinear=multilinear,
            enable_current_limiter=False,
        )
        vc_abc = gfl_inputs[0:3]
        vg_abc = gfl_inputs[3:6]
        i_line_abc = gfl_inputs[6:9]
        v_dc = gfl_inputs[9]
        operating_point_vars = gfl_inputs[10:14]
        d_i_line = [vf.add_diff_var(f"d_{variable.name}", base_var=variable) for variable in i_line_abc]
        d_v_dc = vf.add_diff_var("d_Vdc_", base_var=v_dc)
        i_dc_source = vf.add_var("i_dc_source")
        omega_base = Const(2.0 * np.pi * 50.0)
        r_filter = Const(0.01)
        l_filter = Const(0.10)
        c_dc = Const(10.0)
        filter_rhs = [
            omega_base * (vc - vg - r_filter * current) / l_filter
            for vc, vg, current in zip(vc_abc, vg_abc, i_line_abc)
        ]
        if dc_side == "power_balance":
            dc_rhs = (i_dc_source - p_measured / (v_dc + Const(1.0e-8))) / c_dc
        elif dc_side == "resistance":
            if p0 <= 0.0:
                raise ValueError("dc_side='resistance' requires p0 > 0")
            resistance_dc = Const(vdc0 * vdc0 / p0)
            dc_rhs = (i_dc_source - v_dc / resistance_dc) / c_dc
        else:
            raise ValueError("dc_side must be 'power_balance' or 'resistance'")
        voltage_angle = float(np.angle(voltage))
        voltage_peak = float(np.sqrt(2.0) * abs(voltage))
        i_q0 = 2.0 * p0 / voltage_peak
        i_d0 = -2.0 * q0 / voltage_peak
        phase_angles = (voltage_angle, voltage_angle - 2.0 * np.pi / 3.0,
                        voltage_angle + 2.0 * np.pi / 3.0)
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

        def park(values):
            a, b, c = values
            cosine, sine = np.cos(voltage_angle), np.sin(voltage_angle)
            d_value = (
                2.0 * cosine * a
                + (-cosine + np.sqrt(3.0) * sine) * b
                + (-cosine - np.sqrt(3.0) * sine) * c
            ) / 3.0
            q_value = (
                2.0 * sine * a
                + (-sine - np.sqrt(3.0) * cosine) * b
                + (-sine + np.sqrt(3.0) * cosine) * c
            ) / 3.0
            return d_value, q_value

        vg_d0, vg_q0 = park(grid_voltage_initial)
        vc_d0, vc_q0 = park(converter_voltage_initial)
        vd_hat0 = vc_d0 - (vg_d0 + 0.10 * i_q0)
        vq_hat0 = vc_q0 - (vg_q0 - 0.10 * i_d0)
        initial_values = {
            "P": p0,
            "Q": q0,
            "theta": voltage_angle,
            "omega": 1.0,
            "xi_PLL": 0.0,
            "u_PLL_pi": 0.0,
            "vg_d": vg_d0,
            "vg_q": vg_q0,
            "i_line_d": i_d0,
            "i_line_q": i_q0,
            "i_q_ref": i_q0,
            "i_d_ref": i_d0,
            "xi_Pac_ctrl": i_q0,
            "xi_Qac_ctrl": i_d0,
            "vc_A": converter_voltage_initial[0],
            "vc_B": converter_voltage_initial[1],
            "vc_C": converter_voltage_initial[2],
            "vc_d": vc_d0,
            "vc_q": vc_q0,
            "v_d_c": vc_d0,
            "v_q_c": vc_q0,
            "y_vd_hat": vd_hat0,
            "y_vq_hat": vq_hat0,
            "xi_vd_hat": vd_hat0,
            "xi_vq_hat": vq_hat0,
        }
        if multilinear:
            initial_values["u_cos"] = np.cos(voltage_angle)
            initial_values["u_sin"] = np.sin(voltage_angle)

        # vg_ABC and the DC-source current are the only electrical inputs.
        # vc_ABC is solved by the controller, while i_line_ABC and Vdc are
        # dynamic plant states and observable outputs.
        block = Block(
            name="standalone_gfl",
            in_vars=vg_abc + [i_dc_source],
            out_vars=i_line_abc + [v_dc],
            algebraic_vars=gfl_inputs[:3],
            state_vars=i_line_abc + [v_dc],
            diff_vars=d_i_line + [d_v_dc],
            state_eqs=filter_rhs + [dc_rhs],
            init_eqs={
                i_line_abc[0]: Const(current_initial[0]),
                i_line_abc[1]: Const(current_initial[1]),
                i_line_abc[2]: Const(current_initial[2]),
                v_dc: Const(vdc0),
            },
            diff_init_eqs={
                d_i_line[0]: Const(current_derivative_initial[0]),
                d_i_line[1]: Const(current_derivative_initial[1]),
                d_i_line[2]: Const(current_derivative_initial[2]),
                d_v_dc: Const(0.0),
            },
            event_dict={
                operating_point_vars[0]: Const(p0),
                operating_point_vars[1]: Const(q0),
                operating_point_vars[2]: Const(voltage_peak),
                operating_point_vars[3]: Const(voltage_angle),
            },
            children=[controller],
        )
        block.unify_blocks()
    else:
        raise ValueError("kind must be 'gfm' or 'gfl'")

    phase_voltage = balanced_abc(
        time,
        peak=np.sqrt(2.0) * abs(voltage),
        angle=float(np.angle(voltage)),
        noise_rms=voltage_noise_rms,
        noise_kind=voltage_noise_kind,
    )
    inputs = {}
    for variable in block.in_vars:
        name = variable.name.lower()
        if "v_dc" in name or "vdc" in name:
            inputs[variable] = Const(vdc0)
        elif "vpk_ref" in name:
            inputs[variable] = Const(np.sqrt(2.0))
        elif "i_line" in name:
            inputs[variable] = Const(0.0)
        elif "i_dc_source" in name:
            inputs[variable] = Const(p0 / vdc0)
        elif name.endswith("_a") or "_a_" in name:
            inputs[variable] = phase_voltage[0]
        elif name.endswith("_b") or "_b_" in name:
            inputs[variable] = phase_voltage[1]
        elif name.endswith("_c") or "_c_" in name:
            inputs[variable] = phase_voltage[2]
        else:
            inputs[variable] = Const(0.0)

    options = EmtOptions(
        time_step=1.0e-5,
        simulation_time=simulation_time,
        # The GFL's coupled Park/filter Jacobian is full-rank numerically, but
        # the legacy symbolic Jacobian misses that rank. AD evaluates it correctly.
        solver_type=(EmtSolverTypes.Automatic if kind.lower() == "gfl"
                     else EmtSolverTypes.Symbolic),
        integration_method=DynamicIntegrationMethod.DaeBackEuler,
        initialization_method=EmtInitializationMethod.Explicit,
    )
    runtime_parameters = {}
    noisy_runtime_parameters = {}
    for variable, expression in block.event_dict.items():
        lower_name = variable.name.lower()
        gfl_gain_overrides = {
            "kp_pol": 0.05,
            "ki_pol": 1.0,
            "kp_icl": 0.05,
            "ki_icl": 1.0,
            "kp_pll": 0.01,
            "ki_pll": 0.05,
        }
        if kind.lower() == "gfl" and lower_name in gfl_gain_overrides:
            runtime_parameters[variable] = Const(gfl_gain_overrides[lower_name])
            continue
        if kind.lower() == "gfl" and lower_name == "i_max":
            operating_current = np.sqrt(2.0) * np.hypot(p0, q0) / abs(voltage)
            runtime_parameters[variable] = Const(max(1.2, 1.2 * operating_current))
            continue
        if isinstance(expression, Const) and expression.value is None:
            if "vpk_ref" in lower_name:
                runtime_parameters[variable] = Const(np.sqrt(2.0))
            elif "v_ref" in lower_name:
                runtime_parameters[variable] = Const(1.0)
            else:
                base = p0 if "p_ref" in lower_name else q0
                rms = p_noise_rms if "p_ref" in lower_name else q_noise_rms
                runtime_parameters[variable] = Const(base)
                if rms > 0.0:
                    noisy_runtime_parameters[variable] = (
                        Const(base)
                        + heaviside(time - Const(NOISE_START))
                        * Const(np.sqrt(12.0) * rms)
                        * (rand(time) - Const(0.5))
                    )
    problem = EmtModelProblem(
        block,
        inputs=inputs,
        runtime_parameters=runtime_parameters,
        initial_values=initial_values,
        options=options,
        glob_time=time,
    )
    for variable, expression in noisy_runtime_parameters.items():
        problem.set_runtime_expression(variable, expression)
    return problem


def runtime_input_traces(problem: EmtModelProblem, time: np.ndarray) -> dict[str, np.ndarray]:
    """Replay the seeded symbolic noise and return every prescribed input."""
    np.random.seed(NOISE_SEED)
    variables = list(problem.sys_block.in_vars)
    traces = {variable.name: np.zeros(len(time)) for variable in variables}
    values = problem.event_params_values.copy()
    for sample_index, sample in enumerate(time):
        values = problem.def_event_params_fn(values, float(sample))
        for variable in variables:
            traces[variable.name][sample_index] = values[problem.uid2idx_event_params[variable.uid]]
    return traces


def plot_gfm(problem: EmtModelProblem, result) -> None:
    """Plot noisy terminal voltages and representative GFM responses."""
    figure, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for name, trace in runtime_input_traces(problem, result.time).items():
        axes[0].plot(result.time, trace, label=name)
    axes[0].set_ylabel("Voltage (pu)")
    axes[0].set_title(
        f"Standalone multilinear GFM; {AC_VOLTAGE_NOISE_RMS_PU:.1%} RMS symbolic white voltage noise"
    )
    axes[0].legend(ncol=3)

    for axis, variable_name, label in zip(
        axes[1:],
        ("omega_standalone_gfm", "P_standalone_gfm", "Q_standalone_gfm"),
        ("Frequency (pu)", "Active power (pu)", "Reactive power (pu)"),
    ):
        axis.plot(result.time, problem.trace(result, variable_name))
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)


def main() -> None:
    for kind in ("gfm", "gfl"):
        multilinear = True
        problem = build_problem(kind, multilinear=multilinear)
        try:
            np.random.seed(NOISE_SEED)
            result = problem.simulate()
        except (IndexError, ValueError) as error:
            print(f"{kind.upper()} standalone topology is not square: {error}")
        else:
            print(
                f"{kind.upper()} (multilinear={multilinear}): samples={len(result.time)}, "
                f"initialized={result.initialized}, converged={result.converged}"
            )
            if kind == "gfm":
                plot_gfm(problem, result)
                print(f"plot={OUTPUT}")
                print(
                    f"inputs=balanced 50 Hz ABC voltage, peak={np.sqrt(2.0):.6f} pu; "
                    f"native symbolic uniform white noise={AC_VOLTAGE_NOISE_RMS_PU:.4f} pu RMS/phase, "
                    f"seed={NOISE_SEED}"
                )



if __name__ == "__main__":
    main()
