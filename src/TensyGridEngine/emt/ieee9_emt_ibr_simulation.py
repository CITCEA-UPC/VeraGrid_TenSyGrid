"""IEEE9 EMT study with configurable GFL and GFM generator replacements.

``N_GFL`` and ``N_GFM`` select replacements among the two non-slack IEEE9
generators, in bus order.  Remaining non-slack units use the validated
multilinear Sauer-Pai model.  The inverter builders are the combined/internal-
filter models exercised by the Deliverable 3.3 validation scripts.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.emt_problem_factory import build_emt_problem
from VeraGridEngine.Simulations.EMT.initialization_emt import run_emt_native_initialization
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Templates.Emt.vsc_gfl_emt import (
    VscGflEmtBuild,
    install_gfl_generator_initialization,
)
from VeraGridEngine.Templates.Emt.vsc_gfl_internal_filter_multilinear import (
    add_gfl_internal_filter_multilinear,
)
from VeraGridEngine.Templates.Emt.vsc_gfl_internal_filter import (
    add_gfl_internal_filter,
    connect_gfl_internal_filter_ports,
)
from VeraGridEngine.Templates.Emt.emt_gfm_converter_multilinear import (
    make_gfm_trigonometry_multilinear,
)
from VeraGridEngine.Templates.Emt import emt_gfm_upc
from VeraGridEngine.Utils.Symbolic.block import Block, Var, find_name_in_block
import VeraGridEngine.Utils.Symbolic.symbolic as sym
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import (
    ConverterControlType,
    EmtInitializationMethod,
    EmtProblemTypes,
    EmtSolverTypes,
    VarPowerFlowReferenceType,
)

from emt_ieee9 import (
    attach_emt_models,
    build_emt_options,
    build_ieee9_grid,
    build_power_flow_options,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EROOTS_ROOT = REPO_ROOT.parent
VERAGRID_VALIDATION = EROOTS_ROOT / "VeraGrid" / "trunk" / "dynamics" / "model_validation"

N_GFL = int(os.environ.get("VERAGRID_IEEE9_N_GFL", "1"))
N_GFM = int(os.environ.get("VERAGRID_IEEE9_N_GFM", "0"))
TIME_STEP = float(os.environ.get("VERAGRID_IEEE9_IBR_TIME_STEP", "5e-6"))
SIMULATION_TIME = float(os.environ.get("VERAGRID_IEEE9_IBR_SIM_TIME", "0.001"))
MULTILINEAR_INVERTERS = os.environ.get("VERAGRID_IEEE9_IBR_MULTILINEAR", "0") == "1"


def _load_module(module_name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_deliverable_models():
    # The detailed Colib-aligned GFM is now a proper VeraGrid EMT template.
    gfm = emt_gfm_upc
    if str(VERAGRID_VALIDATION) not in sys.path:
        sys.path.insert(0, str(VERAGRID_VALIDATION))
    gfm_validation = _load_module(
        "ieee9_deliverable_gfm_initialization",
        VERAGRID_VALIDATION / "emt_gfm_model_validation.py",
    )
    gfl = _load_module(
        "ieee9_veragrid_gfl_internal_filter",
        VERAGRID_VALIDATION / "gfl_vsc_emt_internal_filter_validation.py",
    )
    # The combined GFM builder suffixes every variable with its instance name,
    # while the validation initializer asks for unsuffixed logical names.  Its
    # original exact-name lookup therefore silently skips nearly all seeds.
    def find_combined_name(name: str, block: Block) -> Var | None:
        variable = find_name_in_block(name, block)
        if variable is not None:
            return variable
        variable = next(
            (var for var in block.get_all_vars() if var.name.startswith(f"{name}_")),
            None,
        )
        if variable is not None:
            return variable
        return next(
            (
                var
                for owner in block.get_all_blocks()
                for var in (*owner.event_dict.keys(), *owner.parameters.keys())
                if var.name == name or var.name.startswith(f"{name}_")
            ),
            None,
        )

    gfm_validation.find_name_in_block = find_combined_name
    return gfl, gfm, gfm_validation


def _get_signal(problem: EmtProblemDae, values: np.ndarray, block: Block, name: str) -> np.ndarray | None:
    variable: Var | None = find_name_in_block(name, block)
    if variable is None:
        # Imported combined models suffix their variables with the block name.
        # Require ``P_...`` rather than a loose ``P...`` match, which otherwise
        # returns Pt_vsc instead of the controller's measured P.
        variable = next(
            (var for var in block.get_all_vars() if var.name.startswith(f"{name}_")),
            None,
        )
    if variable is None or variable.uid not in problem.uid2idx_vars:
        return None
    return values[:, int(problem.get_var_idx(variable))]


def _set_combined_event_value(block: Block, vf, logical_name: str, value: float) -> None:
    """Set an event parameter by logical or instance-suffixed name."""
    for owner in block.get_all_blocks():
        for variable in list(owner.event_dict):
            if variable.name == logical_name or variable.name.startswith(f"{logical_name}_"):
                owner.event_dict[variable] = vf.add_const(float(value))
                return
    raise KeyError(f"Missing combined-model event parameter {logical_name!r}")


def _find_combined_variable(block: Block, logical_name: str) -> Var | None:
    variable = find_name_in_block(logical_name, block)
    if variable is not None:
        return variable
    return next(
        (
            var for var in block.get_all_vars()
            if var.name.startswith(f"{logical_name}_")
        ),
        None,
    )


def _find_combined_parameter(block: Block, logical_name: str) -> Var | None:
    """Find an event/constant parameter, including combined-model suffixes."""
    for owner in block.get_all_blocks():
        for variable in list(owner.event_dict) + list(owner.parameters):
            if variable.name == logical_name or variable.name.startswith(f"{logical_name}_"):
                return variable
    return _find_combined_variable(block, logical_name)


def _adapt_gfl_as_generator(block: Block, vf) -> None:
    """Close the stiff DC port and adapt VSC branch current to generator injection."""
    vdc = _find_combined_variable(block, "Vdc_")
    if vdc is None:
        raise RuntimeError("GFL model has no Vdc_ input")
    for owner in block.get_all_blocks():
        owner.in_vars = [variable for variable in owner.in_vars if variable.uid != vdc.uid]
    if all(variable.uid != vdc.uid for variable in block.algebraic_vars):
        block.algebraic_vars.append(vdc)
        block.algebraic_eqs.append(vdc - vf.add_const(1.03))

    vg = [_find_combined_variable(block, name) for name in ("vg_A", "vg_B", "vg_C")]
    if any(variable is None for variable in vg):
        raise RuntimeError("GFL model is missing its AC bus-voltage inputs")
    block.external_mapping[VarPowerFlowReferenceType.v_A] = vg[0]
    block.external_mapping[VarPowerFlowReferenceType.v_B] = vg[1]
    block.external_mapping[VarPowerFlowReferenceType.v_C] = vg[2]

    # The GFL uses the VSC branch convention: i_filter is positive from the AC
    # bus into the converter, hence generated P/Q are negative internally.
    # Generator KCL uses the opposite convention (positive injection into the
    # bus).  Keep the controller convention intact and convert only at the
    # device boundary.
    filter_currents = [
        _find_combined_variable(block, name)
        for name in ("i_filter_A", "i_filter_B", "i_filter_C")
    ]
    if any(variable is None for variable in filter_currents):
        raise RuntimeError("GFL internal-filter currents were not found")
    injection_currents = [vf.add_var(f"i_gfl_inj_{phase}") for phase in "ABC"]
    block.algebraic_vars.extend(injection_currents)
    block.algebraic_eqs.extend(
        injection + filter_current
        for injection, filter_current in zip(injection_currents, filter_currents)
    )
    block.external_mapping[VarPowerFlowReferenceType.i_A] = injection_currents[0]
    block.external_mapping[VarPowerFlowReferenceType.i_B] = injection_currents[1]
    block.external_mapping[VarPowerFlowReferenceType.i_C] = injection_currents[2]
    block.external_mapping.pop(VarPowerFlowReferenceType.Vdc, None)
    block.external_mapping.pop(VarPowerFlowReferenceType.Idc, None)


def _gfm_filter_voltage_reference(p_pu: float, q_pu: float, v_peak: float) -> float:
    """Return the steady LCL-capacitor voltage magnitude used by GFM control."""
    rc = 0.01
    lc = 0.10
    i_d = 2.0 * q_pu / v_peak
    i_q = 2.0 * p_pu / v_peak
    v_f_d = rc * i_d - lc * i_q
    v_f_q = v_peak + rc * i_q + lc * i_d
    return float(np.hypot(v_f_d, v_f_q))


def _synchronize_gfm_state_derivatives(problem, block: Block, helper) -> None:
    """Evaluate every GFM state RHS at the final coupled initial point."""
    bindings = helper.uid_bindings_from_problem(problem)
    for _state_var, diff_var, state_rhs in zip(
        block.state_vars, block.diff_vars, block.state_eqs
    ):
        problem.diff_init_guess[diff_var.uid] = float(state_rhs.eval_uid(bindings))


def _raw_generator_targets(grid: gce.MultiCircuit) -> dict[str, tuple[float, float]]:
    pf = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    pf.run()
    if not bool(pf.results.converged):
        raise RuntimeError("Base IEEE9 power flow failed before IBR replacement")
    targets: dict[str, tuple[float, float]] = {}
    for generator in grid.generators:
        if generator.bus.is_slack:
            continue
        bus_index = grid.buses.index(generator.bus)
        bus_power = complex(pf.results.Sbus[bus_index])
        targets[generator.bus.name] = (float(generator.P), float(bus_power.imag))
    return targets


def build_ibr_grid(n_gfl: int, n_gfm: int, multilinear_inverters: bool = False):
    if n_gfl < 0 or n_gfm < 0 or n_gfl + n_gfm > 2:
        raise ValueError("N_GFL and N_GFM must be non-negative and sum to at most two")

    gfl_validation, gfm_models, gfm_validation = _load_deliverable_models()
    grid = build_ieee9_grid()
    targets = _raw_generator_targets(grid)
    non_slack = sorted(
        (generator for generator in grid.generators if not generator.bus.is_slack),
        key=lambda generator: generator.bus.name,
    )
    gfl_generators = non_slack[:n_gfl]
    gfm_generators = non_slack[n_gfl:n_gfl + n_gfm]
    gfl_bus_names = {generator.bus.name for generator in gfl_generators}
    gfm_bus_names = {generator.bus.name for generator in gfm_generators}

    # Start from the already verified nine-line IEEE9 EMT attachment and swap
    # only the selected generator models.
    attach_emt_models(grid)
    gfm_blocks: list[tuple[str, Block]] = []
    gfl_blocks: list[tuple[Any, Any, Block, tuple[float, float]]] = []

    for replacement_index, generator in enumerate(gfl_generators, start=1):
        bus_name = generator.bus.name
        vf = grid.var_factory
        block = VscGflEmtBuild(
            vf, name=f"emt_gfl_{replacement_index}_{bus_name}",
            control1=ConverterControlType.Pac,
            control2=ConverterControlType.Qac,
            frozen_voltage_source=False,
            multilinear=multilinear_inverters,
        ).block
        p_mw, q_mvar = targets[bus_name]
        operating_s_pu = float(np.hypot(p_mw, q_mvar) / grid.Sbase)
        # dq currents are peak quantities: Ipk_pu = sqrt(2)*|S|/|V|.
        # Give the replacement converter 20% current headroom at 1 p.u.
        # voltage while retaining at least the legacy 1.2-p.u. limit.
        peak_current_limit = max(1.2, 1.2 * np.sqrt(2.0) * operating_s_pu)
        gfl_validation.set_event_value_in_block(vf, "I_max", block, peak_current_limit)
        if multilinear_inverters:
            add_gfl_internal_filter_multilinear(vf, grid.fBase, block)
        else:
            add_gfl_internal_filter(vf, grid.fBase, block)
        gfl_validation.set_event_value_in_block(vf, "R_filter", block, 0.01)
        connect_gfl_internal_filter_ports(vf, block)
        _adapt_gfl_as_generator(block, vf)
        set_emt_model(device=generator, model=block, var_factory=vf)
        # Port connection/substitution performed by set_emt_model may replace
        # converter-side variables, so install initialization on the finalized
        # device block.
        install_gfl_generator_initialization(generator.emt_model, vf, grid.fBase)
        block = generator.emt_model
        gfl_blocks.append((generator, generator.bus, block, targets[bus_name]))

    for replacement_index, generator in enumerate(gfm_generators, start=1):
        bus_name = generator.bus.name
        vf = grid.var_factory
        block = gfm_models.build_emt_gfm_aggregated_model(
            vf=vf,
            name=f"emt_gfm_{replacement_index}_{bus_name}",
            multilinear=multilinear_inverters,
        )
        if multilinear_inverters:
            make_gfm_trigonometry_multilinear(block, vf)
        os.environ.setdefault("VERAGRID_GFM_EMT_KP_VCL", "0.00075")
        os.environ.setdefault("VERAGRID_GFM_EMT_KI_VCL", "0.2")
        os.environ.setdefault("VERAGRID_GFM_EMT_KP_ICL", "0.00075")
        os.environ.setdefault("VERAGRID_GFM_EMT_KI_ICL", "0.2")
        gfm_validation.apply_gfm_gain_overrides(block, vf)
        set_emt_model(device=generator, model=block, var_factory=vf)
        gfm_blocks.append((bus_name, block))

    return grid, gfl_blocks, gfl_bus_names, {name for name, _ in gfm_blocks}, gfm_blocks, gfm_validation


def _seed_gfl_internal_filter(problem, grid, pf_results, entry, helper) -> None:
    """Seed the Deliverable internal-filter GFL at its PF operating point."""
    _generator, ac_bus, model, (p_mw, q_mvar) = entry
    bus_index = grid.buses.index(ac_bus)
    voltage = complex(pf_results.voltage[bus_index])
    vpk = float(np.sqrt(2.0) * abs(voltage))
    angle = float(np.angle(voltage))
    va = vpk * np.sin(angle)
    vb = vpk * np.sin(angle - 2.0 * np.pi / 3.0)
    vc = vpk * np.sin(angle + 2.0 * np.pi / 3.0)

    # For v_a=Vpk*sin(angle), this Park transform has vd=0 and vq>0 at
    # alpha=pi-angle.  Use the exact wrapped angle; a sampled angle search left
    # a small vd error that kicked the PLL and Q controller at the first step.
    theta_ref = float(np.arctan2(np.sin(np.pi - angle), np.cos(np.pi - angle)))
    vd_g, vq_g = helper.park_dq_value(theta_ref, va, vb, vc)
    # Do not infer converter current from a neighbouring branch index: adding
    # a VSC changes the PF branch ordering.  Solve the balanced abc currents
    # directly from the scheduled terminal P/Q and the PF voltage instead.
    # Native VSC branch convention: current and power are positive from the AC
    # bus into the converter, hence generation has negative P/Q.
    p_target = -p_mw / grid.Sbase
    q_target = -q_mvar / grid.Sbase
    current_matrix = np.asarray(
        [
            [va / 3.0, vb / 3.0, vc / 3.0],
            [(vb - vc) / (3.0 * np.sqrt(3.0)),
             (vc - va) / (3.0 * np.sqrt(3.0)),
             (va - vb) / (3.0 * np.sqrt(3.0))],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    ia_filter, ib_filter, ic_filter = np.linalg.solve(
        current_matrix, np.asarray([p_target, q_target, 0.0], dtype=float)
    )
    id_filter, iq_filter = helper.park_dq_value(
        theta_ref, ia_filter, ib_filter, ic_filter
    )
    r_filter = 0.01
    x_filter = 0.10
    # Periodic RL drop for x_dq=T(-theta)x_abc, theta_dot=omega_b*omega,
    # and current positive from the grid bus toward the converter.
    vd_c = vd_g - r_filter * id_filter + x_filter * iq_filter
    vq_c = vq_g - r_filter * iq_filter - x_filter * id_filter
    va_conv, vb_conv, vc_conv = helper.inverse_park_value(theta_ref, vd_c, vq_c)
    p_meas = (va * ia_filter + vb * ib_filter + vc * ic_filter) / 3.0
    q_meas = ((vb - vc) * ia_filter + (vc - va) * ib_filter + (va - vb) * ic_filter) / (3.0 * np.sqrt(3.0))
    vdc0 = 1.03
    p_conv = -(va_conv * ia_filter + vb_conv * ib_filter + vc_conv * ic_filter) / 3.0
    omega_base = 2.0 * np.pi * grid.fBase
    kp_outer = float(os.environ.get("VERAGRID_GFL_EMT_KP_POL", "0.05"))
    ki_outer = float(os.environ.get("VERAGRID_GFL_EMT_KI_POL", "1.0"))
    kp_current = float(os.environ.get("VERAGRID_GFL_EMT_KP_ICL", "0.05"))
    ki_current = float(os.environ.get("VERAGRID_GFL_EMT_KI_ICL", "1.0"))
    kp_pll = float(os.environ.get("VERAGRID_GFL_EMT_KP_PLL", "0.01"))
    ki_pll = float(os.environ.get("VERAGRID_GFL_EMT_KI_PLL", "0.05"))
    dia = omega_base * (va - va_conv - r_filter * ia_filter) / x_filter
    dib = omega_base * (vb - vb_conv - r_filter * ib_filter) / x_filter
    dic = omega_base * (vc - vc_conv - r_filter * ic_filter) / x_filter

    values = {
        "P": p_meas, "Q": q_meas, "P_f": p_meas, "Q_f": q_meas,
        "Pt_vsc": -p_meas, "Qt_vsc": -q_meas,
        "i_a_f": ia_filter, "i_b_f": ib_filter, "i_c_f": ic_filter,
        "i_filter_A": ia_filter, "i_filter_B": ib_filter, "i_filter_C": ic_filter,
        "i_dc": p_conv / vdc0, "i_conv_dc": p_conv / vdc0, "P_conv": p_conv,
        "theta": -theta_ref, "omega": 1.0, "xi_PLL": 0.0,
        "vg_d": vd_g, "vg_q": vq_g, "vc_d": vd_c, "vc_q": vq_c,
        "i_line_d": id_filter, "i_line_q": iq_filter,
        "i_d_ref": id_filter, "i_q_ref": iq_filter,
        "v_d_c": vd_c, "v_q_c": vq_c,
        "va_v": va_conv, "vb_v": vb_conv, "vc_v": vc_conv,
        "u_Pac_ctrl": 0.0, "u_Qac_ctrl": 0.0,
        "y_vd_hat": 0.0, "y_vq_hat": 0.0,
        "u_vd_hat": 0.0, "u_vq_hat": 0.0,
        "Vdc_": vdc0, "Vdc_cap": vdc0,
        "xi_Pac_ctrl": iq_filter / ki_outer,
        "xi_Qac_ctrl": id_filter / ki_outer,
        "xi_vd_hat": 0.0, "xi_vq_hat": 0.0,
        "u_PLL_pi": vd_g,
    }
    for name, value in values.items():
        helper.set_init_if_exists(problem, model, name, value)

    for name, value in (
        ("P_ref", p_meas), ("Q_ref", q_meas), ("Vm_ac_ref", vq_g),
        ("Kp_pol", kp_outer), ("Ki_pol", ki_outer),
        ("Kp_icl", kp_current), ("Ki_icl", ki_current),
        ("Kp_pll", kp_pll), ("Ki_pll", ki_pll),
        ("L", x_filter), ("R_filter", r_filter), ("Cdc", 10.0),
    ):
        helper.set_problem_runtime_value(problem, grid.var_factory, model, name, value)

    theta = helper.find_name_in_block("theta", model)
    if theta is not None and theta.diff_var is not None:
        problem.diff_init_guess[theta.diff_var.uid] = omega_base
    for name, derivative in (
        ("dt_i_filter_A", dia), ("dt_i_filter_B", dib), ("dt_i_filter_C", dic),
    ):
        diff_var = helper.find_name_in_block(name, model)
        if diff_var is not None:
            problem.diff_init_guess[diff_var.uid] = derivative
    for name in ("dt_1_i_q_ref", "dt_1_i_d_ref", "dt_1_u_Pac_ctrl", "dt_1_u_Qac_ctrl"):
        diff_var = helper.find_name_in_block(name, model)
        if diff_var is not None:
            problem.diff_init_guess[diff_var.uid] = 0.0


def _synchronize_gfm_multilinear_trig(problem, block: Block, frequency_hz: float) -> None:
    """Keep GFM trig auxiliaries consistent after the legacy PF state seed."""
    theta = _find_combined_variable(block, "theta")
    omega = _find_combined_variable(block, "omega")
    u_cos = _find_combined_variable(block, "u_cos_gfm")
    u_sin = _find_combined_variable(block, "u_sin_gfm")
    if theta is None or omega is None or u_cos is None or u_sin is None:
        return
    theta0 = float(problem.init_guess[theta.uid])
    omega0 = float(problem.init_guess[omega.uid])
    cos0, sin0 = float(np.cos(theta0)), float(np.sin(theta0))
    problem.init_guess[u_cos.uid] = cos0
    problem.init_guess[u_sin.uid] = sin0
    theta_rate = 2.0 * np.pi * frequency_hz * omega0
    if u_cos.diff_var is not None:
        problem.diff_init_guess[u_cos.diff_var.uid] = -theta_rate * sin0
    if u_sin.diff_var is not None:
        problem.diff_init_guess[u_sin.diff_var.uid] = theta_rate * cos0


def _synchronize_gfl_multilinear_trig(problem, block: Block, helper, frequency_hz: float) -> None:
    """Propagate finalized PLL/auxiliary angles through every GFL trig lift."""
    # Initial equations are evaluated during generic assembly before the custom
    # PF seed overwrites theta. Re-evaluate angle and trigonometric auxiliaries
    # in dependency order at the finalized operating point.
    for _ in range(3):
        bindings = helper.uid_bindings_from_problem(problem)
        for owner in block.get_all_blocks():
            for variable, expression in owner.init_eqs.items():
                if variable.name.startswith(("theta_aux", "u_cos", "u_sin")):
                    problem.init_guess[variable.uid] = float(expression.eval_uid(bindings))

    # This lift mirrors the production filter and therefore uses theta itself.
    # Seed this uniquely named lift explicitly: the combined GFL contains
    # several generic ``u_cos``/``u_sin`` lifts and name-based traversal can
    # otherwise leave this one holding the value of a different Park block.
    theta = _find_combined_variable(block, "theta")
    inverse_angle = _find_combined_variable(block, "theta_aux_filter_inv")
    inverse_cosine = _find_combined_variable(block, "u_cos_filter_inv")
    inverse_sine = _find_combined_variable(block, "u_sin_filter_inv")
    if all(item is not None for item in (theta, inverse_angle, inverse_cosine, inverse_sine)):
        inverse_angle_value = float(problem.init_guess[theta.uid])
        problem.init_guess[inverse_angle.uid] = inverse_angle_value
        problem.init_guess[inverse_cosine.uid] = float(np.cos(inverse_angle_value))
        problem.init_guess[inverse_sine.uid] = float(np.sin(inverse_angle_value))
        if os.environ.get("VERAGRID_IEEE9_IBR_DIAGNOSTICS", "0") == "1":
            print(
                "GFL inverse-Park seed: "
                f"theta={problem.init_guess[theta.uid]:.9e}, "
                f"angle={inverse_angle_value:.9e}, "
                f"cos={problem.init_guess[inverse_cosine.uid]:.9e}, "
                f"sin={problem.init_guess[inverse_sine.uid]:.9e}"
            )

    # symbolic_ml.trig_transform expresses these as algebraic equations in the
    # differential variables, so it has no diff_init_eqs of its own.
    for owner in block.get_all_blocks():
        d_cosines = [v for v in owner.diff_vars if v.name == "d_u_cos"]
        d_sines = [v for v in owner.diff_vars if v.name == "d_u_sin"]
        d_angles = [v for v in owner.diff_vars if v.name == "d_delta"]
        theta = _find_combined_variable(block, "theta")
        omega = _find_combined_variable(block, "omega")
        pll_rate = 2.0 * np.pi * frequency_hz * (
            float(problem.init_guess[omega.uid]) if omega is not None else 1.0
        )
        for d_cos, d_sin, d_angle in zip(d_cosines, d_sines, d_angles):
            cos_var, sin_var = d_cos.base_var, d_sin.base_var
            if cos_var is None or sin_var is None:
                continue
            angle_rate = pll_rate
            problem.diff_init_guess[d_angle.uid] = angle_rate
            cos_value = float(problem.init_guess[cos_var.uid])
            sin_value = float(problem.init_guess[sin_var.uid])
            problem.diff_init_guess[d_cos.uid] = -angle_rate * sin_value
            problem.diff_init_guess[d_sin.uid] = angle_rate * cos_value



def run_case(
    n_gfl: int,
    n_gfm: int,
    multilinear_inverters: bool = False,
    configure_events: Callable[[Any, list, list], None] | None = None,
):
    grid, gfl_devices, gfl_bus_names, gfm_bus_names, gfm_blocks, gfm_validation = build_ibr_grid(
        n_gfl, n_gfm, multilinear_inverters=multilinear_inverters
    )
    pf = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError(
            f"IBR power flow failed: balanced={pf.results.converged}, three_phase={pf3.results.converged}"
        )

    options = build_emt_options()
    # Both Deliverable 3.3 combined-converter validations use StructuralAD.
    # Their internal-filter blocks are assembled dynamically and are not
    # compatible with the static Jacobian coordinate layout of Symbolic.
    options.solver_type = EmtSolverTypes.StructuralAD
    options.time_step = TIME_STEP
    options.simulation_time = SIMULATION_TIME
    # The IEEE9 network remains a mixed formulation: only the selected inverter
    # trig products are multilinearized, while lines, loads and any remaining
    # machines still use the current-balance DAE contract.  Selecting the global
    # Multilinear problem compiler for this hybrid grid drops/reshapes equations
    # and produces a spurious first-step jump.  Both inverter formulations must
    # therefore be compared through the same CurrentBalance assembly.
    options.problem_type = EmtProblemTypes.CurrentBalance
    buses_by_name = {bus.name: bus for bus in grid.buses}
    # Materialize the GFM controller references before EmtProblemDae clones
    # and compiles the device blocks.  Updating the original event_dict after
    # problem construction is too late for parameters initialized from None.
    for bus_name, gfm_block in gfm_blocks:
        bus_index = grid.buses.index(buses_by_name[bus_name])
        p_reference = float(np.real(pf.results.Sbus[bus_index]) / grid.Sbase)
        q_reference = float(np.imag(pf.results.Sbus[bus_index]) / grid.Sbase)
        v_peak = float(np.sqrt(2.0) * abs(pf.results.voltage[bus_index]))
        for logical_name, value in (
            ("P_ref", p_reference),
            ("Q_ref", q_reference),
            ("V_ref", _gfm_filter_voltage_reference(p_reference, q_reference, v_peak)),
        ):
            _set_combined_event_value(
                gfm_block, grid.var_factory, logical_name, value
            )
    if configure_events is not None:
        configure_events(grid, gfl_devices, gfm_blocks)
    problem = build_emt_problem(
        grid=grid,
        options=options,
        pf_results=pf.results,
        pf_results_3ph=pf3.results,
    )
    generators_by_bus = {generator.bus.name: generator for generator in grid.generators}
    for bus_name, gfm_block in gfm_blocks:
        gfm_validation.seed_gfm_from_power_flow(
            problem, grid, pf.results, gfm_block, buses_by_name[bus_name]
        )
        if multilinear_inverters:
            _synchronize_gfm_multilinear_trig(problem, gfm_block, grid.fBase)
        bus_index = grid.buses.index(buses_by_name[bus_name])
        p_ref = float(np.real(pf.results.Sbus[bus_index]) / grid.Sbase)
        q_ref = float(np.imag(pf.results.Sbus[bus_index]) / grid.Sbase)
        vq_f = gfm_validation.find_name_in_block("vq_f", gfm_block)
        v_ref = (
            float(problem.init_guess[vq_f.uid])
            if vq_f is not None and vq_f.uid in problem.init_guess
            else float(np.sqrt(2.0) * abs(pf.results.voltage[bus_index]))
        )
        working_block = problem._working_emt_models[str(generators_by_bus[bus_name].idtag)]
        for logical_name, value in (
            ("P_ref", p_ref), ("Q_ref", q_ref), ("V_ref", v_ref),
        ):
            original_var = gfm_validation.find_name_in_block(logical_name, gfm_block)
            if original_var is not None:
                problem.set_internal_runtime_if_exists(
                    working_block, original_var.name, value
                )
        theta = find_name_in_block("theta", gfm_block)
        if theta is None:
            theta = next(
                (var for var in gfm_block.get_all_vars() if var.name.startswith("theta_")),
                None,
            )
        if theta is not None and theta.diff_var is not None:
            problem.diff_init_guess[theta.diff_var.uid] = -2.0 * np.pi * grid.fBase
        _synchronize_gfm_state_derivatives(problem, gfm_block, gfm_validation)
    if os.environ.get("VERAGRID_IEEE9_IBR_DIAGNOSTICS", "0") == "1":
        bindings = gfm_validation.uid_bindings_from_problem(problem)
        print("Machine/exciter assembled initial values:")
        diagnostic_prefixes = ("Vf", "IRPu", "u_aux", "exp_", "AEx", "BEx", "Se_threshold", "y_exciter4")
        for variable in problem.get_state_vars() + problem.get_algebraic_vars() + problem.get_variable_parameters():
            if variable.name.startswith(diagnostic_prefixes) and variable.uid in bindings:
                print(f"  {variable.name}: {float(bindings[variable.uid]): .9e}")
        print("Machine/exciter surviving init equations:")
        variable_by_uid = {
            variable.uid: variable
            for variable in problem.get_state_vars() + problem.get_algebraic_vars()
        }
        for variable, expression in problem.sys_block.init_eqs.items():
            if variable.name.startswith(("Vf", "IRPu", "exp_", "y_exciter4", "u_aux")):
                target = variable_by_uid.get(variable.uid, variable)
                try:
                    evaluated = float(expression.eval_uid(bindings))
                except (KeyError, ValueError):
                    evaluated = float("nan")
                print(
                    f"  {target.name}[{target.uid}] = {expression}; "
                    f"eval={evaluated:.9e}"
                )
    if gfm_blocks:
        gfm_validation.seed_bus_algebraic_predictors(problem, grid, pf.results)
        if os.environ.get("VERAGRID_IEEE9_IBR_DIAGNOSTICS", "0") == "1":
            for _bus_name, gfm_block in gfm_blocks:
                for diagnostic_name in (
                    "id_ref", "id_c", "iq_ref", "iq_c", "z_id_loop", "z_iq_loop",
                    "vd_ctrl_out", "vq_ctrl_out", "Kp_icl", "Ki_icl",
                ):
                    diagnostic_var = gfm_validation.find_name_in_block(
                        diagnostic_name, gfm_block
                    )
                    if diagnostic_var is None:
                        continue
                    if diagnostic_var.uid in problem.uid2idx_vars:
                        diagnostic_value = problem.init_guess[diagnostic_var.uid]
                    elif diagnostic_var.uid in problem.uid2idx_event_params:
                        diagnostic_value = problem.event_params_values[
                            problem.uid2idx_event_params[diagnostic_var.uid]
                        ]
                    else:
                        continue
                    print(f"GFM init {diagnostic_name}={diagnostic_value:.9e}")
                gfm_validation.print_gfm_algebraic_residuals(problem, gfm_block)
                gfm_validation.print_gfm_state_residuals(problem, gfm_block)
            gfm_validation.print_global_state_residuals(problem)
            gfm_validation.print_bus_kcl_residuals(problem, grid)
    if multilinear_inverters:
        for _generator, _bus, gfl_block, _target in gfl_devices:
            _synchronize_gfl_multilinear_trig(
                problem, gfl_block, gfm_validation, grid.fBase
            )
    if gfl_devices:
        # GFL internal states are initialized by the model's symbolic init_eqs.
        # The IEEE9 machine/network seed is already a periodic EMT operating
        # point. A global algebraic Newton pass moves that orbit while resolving
        # unrelated exciter auxiliary variables, so keep it opt-in.
        if os.environ.get("VERAGRID_IEEE9_GFL_POST_INIT", "0") != "0":
            previous_initialization_method = options.initialization_method
            previous_initialization_tolerance = options.init_newton_tol
            options.initialization_method = EmtInitializationMethod.ConsistentNewton
            options.init_newton_tol = options.tolerance
            init_report = run_emt_native_initialization(problem=problem, options=options)
            options.initialization_method = previous_initialization_method
            options.init_newton_tol = previous_initialization_tolerance
            print(
                "GFL post-seed coupled initialization: "
                f"status={init_report.status.name}, used={init_report.method_used.name}, "
                f"res0={init_report.initial_residual_inf:.6e}, "
                f"resf={init_report.final_residual_inf:.6e}"
            )
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=SIMULATION_TIME,
        h=TIME_STEP,
        method=options.integration_method,
    )
    x0_seed = problem.get_x0().copy()
    dx0_seed = problem.get_dx0().copy()
    time, values, _derivatives, initialized, converged = solver.simulate(
        boundary_updater=cast(Any, problem)
    )
    print(
        f"N_GFL={n_gfl}, N_GFM={n_gfm}, pf=True, pf3=True, "
        f"initialized={initialized}, converged={converged}, steps={len(time)-1}"
    )
    for bus_name, gfm_block in gfm_blocks:
        diagnostics = []
        for logical_name in ("omega", "P", "y_p_lp", "P_ref", "Kdp"):
            var = gfm_validation.find_name_in_block(logical_name, gfm_block)
            if var is None:
                continue
            if var.uid in problem.uid2idx_vars:
                trace = values[:, int(problem.get_var_idx(var))]
                diagnostics.append(f"{logical_name}={trace[0]:.9g}->{trace[-1]:.9g}")
            elif var.uid in problem.uid2idx_event_params:
                idx = int(problem.uid2idx_event_params[var.uid])
                diagnostics.append(f"{logical_name}={problem.event_params_values[idx]:.9g}")
        print(f"GFM {bus_name}: " + ", ".join(diagnostics))
    for _generator, ac_bus, gfl_block, _target in gfl_devices:
        diagnostics = []
        for logical_name in ("omega", "P", "Q", "Vdc_cap"):
            trace = _get_signal(problem, values, gfl_block, logical_name)
            if trace is None:
                continue
            diagnostics.append(
                f"{logical_name}={trace[0]:.9g}->{trace[-1]:.9g} "
                f"(delta={trace[-1] - trace[0]:+.3e})"
            )
        print(f"GFL {ac_bus.name}: " + ", ".join(diagnostics))
    if gfl_devices and (
        not initialized or not converged
        or os.environ.get("VERAGRID_IEEE9_IBR_DIAGNOSTICS", "0") == "1"
    ):
        gfl_helper, _gfm_models, _gfm_validation = _load_deliverable_models()
        gfl_helper.print_initialization_diagnostics(
            problem,
            gfl_devices[0][2],
            np.asarray(time),
            np.asarray(values),
            np.asarray(_derivatives),
            x0_seed,
            dx0_seed,
        )
    return problem, np.asarray(time), np.asarray(values), gfl_devices, gfl_bus_names, gfm_bus_names


def main() -> None:
    problem, time, values, gfl_devices, gfl_bus_names, gfm_bus_names = run_case(
        N_GFL, N_GFM, multilinear_inverters=MULTILINEAR_INVERTERS
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for generator in problem.grid.generators:
        if generator.bus.name not in gfm_bus_names:
            continue
        for signal_name, axis in (("omega", axes[0]), ("P", axes[1]), ("Q", axes[2])):
            trace = _get_signal(problem, values, generator.emt_model, signal_name)
            if trace is not None:
                axis.plot(1e3 * time, trace, label=f"GFM {generator.bus.name}")
    for _generator, ac_bus, model, _target in gfl_devices:
        for signal_name, axis in (("omega", axes[0]), ("P", axes[1]), ("Q", axes[2])):
            trace = _get_signal(problem, values, model, signal_name)
            if trace is not None:
                axis.plot(1e3 * time, trace, label=f"GFL {ac_bus.name}")
    axes[0].set_ylabel("omega (p.u.)")
    axes[1].set_ylabel("P (p.u.)")
    axes[2].set_ylabel("Q (p.u.)")
    axes[2].set_xlabel("Time (ms)")
    axes[0].set_title(f"IEEE9 EMT IBR replacements: N_GFL={N_GFL}, N_GFM={N_GFM}")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend()
    output = Path(__file__).with_name("results") / f"ieee9_emt_ibr_gfl{N_GFL}_gfm{N_GFM}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"plot={output}")


if __name__ == "__main__":
    main()
