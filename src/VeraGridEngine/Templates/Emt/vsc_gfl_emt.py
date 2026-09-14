# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import List
import numpy as np
import math

from VeraGridEngine.enumerations import DeviceType
from VeraGridEngine.Devices.Dynamic.emt_template import EmtModelTemplate
from VeraGridEngine.Utils.Symbolic.block import (Block, Var, VarPowerFlowReferenceType, find_name_in_block)
from VeraGridEngine.Utils.Symbolic.block_helpers import tf_to_block
from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
import VeraGridEngine.Utils.Symbolic.symbolic as sym
import VeraGridEngine.Utils.Symbolic.symbolic_ml as symbolic_ml
from VeraGridEngine.enumerations import ConverterControlType


def _find_model_var(block: Block, logical_name: str) -> Var | None:
    """Find an exact or combined-model variable by its logical name."""
    variable = find_name_in_block(logical_name, block)
    if variable is not None:
        return variable
    return next(
        (item for item in block.get_all_vars()
         if item.name.startswith(f"{logical_name}_")),
        None,
    )


def _find_model_parameter(block: Block, logical_name: str) -> Var | None:
    """Find an event/constant parameter, including combined-model suffixes."""
    for owner in block.get_all_blocks():
        for variable in (*owner.event_dict.keys(), *owner.parameters.keys()):
            if (variable.name == logical_name
                    or variable.name.startswith(f"{logical_name}_")):
                return variable
    return _find_model_var(block, logical_name)


def install_gfl_generator_initialization(
        block: Block,
        vfactory: VarFactory,
        frequency_hz: float,
) -> None:
    """Install the complete symbolic periodic initialization of a GFL generator.

    The equations depend only on PF-mapped symbolic parameters and model
    parameters.  No simulation script needs to calculate or write numerical
    state/algebraic guesses.
    """
    for owner in block.get_all_blocks():
        owner.init_eqs = {
            variable: equation for variable, equation in owner.init_eqs.items()
            if getattr(variable, "name", None) not in {"vc_d", "vc_q"}
        }

    get = lambda name: _find_model_var(block, name)
    p, q = get("P"), get("Q")
    pt_vsc, qt_vsc = get("Pt_vsc"), get("Qt_vsc")
    p_ref = _find_model_parameter(block, "P_ref")
    q_ref = _find_model_parameter(block, "Q_ref")
    vpk_ref = _find_model_parameter(block, "Vpk_ref")
    phi_ref = _find_model_parameter(block, "phi_v_ref")
    theta, omega = get("theta"), get("omega")
    vgd, vgq = get("vg_d"), get("vg_q")
    id_line, iq_line = get("i_line_d"), get("i_line_q")
    id_ref, iq_ref = get("i_d_ref"), get("i_q_ref")
    vdc, vdc_cap = get("Vdc_"), get("Vdc_cap")
    vd_c, vq_c = get("v_d_c"), get("v_q_c")
    yvd, yvq = get("y_vd_hat"), get("y_vq_hat")
    resistance = _find_model_parameter(block, "R_filter")
    reactance = _find_model_parameter(block, "L")
    filter_i = [get(f"i_filter_{phase}") for phase in "ABC"]
    injection_i = [get(f"i_gfl_inj_{phase}") for phase in "ABC"]
    converter_v = [get(name) for name in ("va_v", "vb_v", "vc_v")]
    bus_v = [
        get(name) or block.external_mapping.get(reference)
        for name, reference in zip(
            ("vg_A", "vg_B", "vg_C"),
            (VarPowerFlowReferenceType.v_A, VarPowerFlowReferenceType.v_B,
             VarPowerFlowReferenceType.v_C),
        )
    ]
    required = [p, q, pt_vsc, qt_vsc, p_ref, q_ref, vpk_ref, phi_ref,
                theta, omega, vgd, vgq, id_line, iq_line, id_ref, iq_ref,
                vdc, vdc_cap, vd_c, vq_c, yvd, yvq, resistance, reactance,
                *filter_i, *injection_i, *converter_v, *bus_v]
    if any(item is None for item in required):
        raise RuntimeError("Incomplete GFL generator initialization interface")

    p_pf = vfactory.add_var("P_gfl_generator_pf")
    q_pf = vfactory.add_var("Q_gfl_generator_pf")
    block.event_dict[p_pf] = vfactory.add_const(None)
    block.event_dict[q_pf] = vfactory.add_const(None)
    block.external_mapping[VarPowerFlowReferenceType.P] = p_pf
    block.external_mapping[VarPowerFlowReferenceType.Q] = q_pf

    zero = vfactory.add_const(0.0)
    one = vfactory.add_const(1.0)
    two = vfactory.add_const(2.0)
    half = vfactory.add_const(0.5)
    sqrt3half = vfactory.add_const(np.sqrt(3.0) / 2.0)
    cosine, sine = sym.cos(theta), sym.sin(theta)
    id0 = two * q_pf / vpk_ref
    iq0 = -two * p_pf / vpk_ref
    ia0 = id0 * cosine + iq0 * sine
    ib0 = id0 * (-half * cosine + sqrt3half * sine) + iq0 * (-half * sine - sqrt3half * cosine)
    ic0 = id0 * (-half * cosine - sqrt3half * sine) + iq0 * (-half * sine + sqrt3half * cosine)
    vd0 = -resistance * id0 - reactance * omega * iq0
    vq0 = vpk_ref - resistance * iq0 + reactance * omega * id0
    va0 = vd0 * cosine + vq0 * sine
    vb0 = vd0 * (-half * cosine + sqrt3half * sine) + vq0 * (-half * sine - sqrt3half * cosine)
    vc0 = vd0 * (-half * cosine - sqrt3half * sine) + vq0 * (-half * sine + sqrt3half * cosine)

    block.init_eqs.update({
        p: -p_pf, q: -q_pf, pt_vsc: p_pf, qt_vsc: q_pf,
        p_ref: -p_pf, q_ref: -q_pf,
        theta: phi_ref, omega: one, vgd: zero, vgq: vpk_ref,
        id_line: id0, iq_line: iq0, id_ref: id0, iq_ref: iq0,
        yvd: zero, yvq: zero, vd_c: vd0, vq_c: vq0,
        converter_v[0]: va0, converter_v[1]: vb0, converter_v[2]: vc0,
        filter_i[0]: ia0, filter_i[1]: ib0, filter_i[2]: ic0,
        injection_i[0]: -ia0, injection_i[1]: -ib0, injection_i[2]: -ic0,
        vdc: vfactory.add_const(1.03), vdc_cap: vdc,
    })

    omega_base = vfactory.add_const(2.0 * np.pi * frequency_hz)
    if theta.diff_var is not None:
        block.diff_init_eqs[theta.diff_var] = omega_base
    for current, voltage_bus, voltage_converter in zip(filter_i, bus_v, converter_v):
        if current.diff_var is not None:
            block.diff_init_eqs[current.diff_var] = omega_base * (
                voltage_bus - voltage_converter - resistance * current
            ) / reactance
    for name in ("dt_1_i_q_ref", "dt_1_i_d_ref", "dt_1_u_Pac_ctrl",
                 "dt_1_u_Qac_ctrl", "dt_1_y_vd_hat", "dt_1_y_vq_hat",
                 "dt_1_u_vd_hat", "dt_1_u_vq_hat"):
        variable = get(name)
        if variable is not None:
            block.diff_init_eqs[variable] = zero


def inverse_park_transform_block(vfactory: VarFactory, v_dq: list[Var], theta: Var, aux_vars = None, multilinear: bool = False, name:str =''):
    """
    Create a symbolic inverse Park transform (dq → abc) block for voltages.

    v_dq = [v_d_c, v_q_c]
    theta : grid angle (from PLL)
    multilinear : if True, use u_cos/u_sin auxiliary variables (for multilinearization)
    """

    v_d_c, v_q_c = v_dq

    # abc variables
    va_c = Var(name[0]+'a_'+name[-1])
    vb_c = Var(name[0]+'b_'+name[-1])
    vc_c = Var(name[0]+'c_'+name[-1])

    sqrt3 = vfactory.add_const(np.sqrt(3))
    reformulated_vars = []
    if not multilinear:
        # Non-multilinear form (explicit cos/sin)
        algebraic_eqs = [
            va_c - (v_d_c * sym.cos(theta) + v_q_c * sym.sin(theta)),

            vb_c - ((-vfactory.add_const(0.5) * v_d_c - (sqrt3 / vfactory.add_const(2)) * v_q_c) * sym.cos(theta)
                    + ((sqrt3 / vfactory.add_const(2)) * v_d_c - vfactory.add_const(0.5) * v_q_c) * sym.sin(theta)),

            vc_c - ((-vfactory.add_const(0.5) * v_d_c + (sqrt3 / vfactory.add_const(2)) * v_q_c) * sym.cos(theta)
                    + ((-sqrt3 / vfactory.add_const(2)) * v_d_c - vfactory.add_const(0.5) * v_q_c) * sym.sin(theta)),
        ]
        algebraic_vars = [va_c, vb_c, vc_c]
        trig_block = Block()
        aux_vars = None
    else:
        trig_block, u_cos, u_sin = symbolic_ml.trig_transform(vfactory, theta, type="usual")
        algebraic_eqs = [
            va_c - (v_d_c * u_cos + v_q_c * u_sin),
            vb_c - ((-vfactory.add_const(0.5) * v_d_c - (sqrt3 / vfactory.add_const(2)) * v_q_c) * u_cos
                    + ((sqrt3 / vfactory.add_const(2)) * v_d_c - vfactory.add_const(0.5) * v_q_c) * u_sin),
            vc_c - ((-vfactory.add_const(0.5) * v_d_c + (sqrt3 / vfactory.add_const(2)) * v_q_c) * u_cos
                    + ((-sqrt3 / vfactory.add_const(2)) * v_d_c - vfactory.add_const(0.5) * v_q_c) * u_sin),
        ]
        algebraic_vars = [va_c, vb_c, vc_c]
        reformulated_vars = [u_sin, u_cos]
        aux_vars = (u_cos, u_sin)

    inv_park_block = Block(
        algebraic_eqs=algebraic_eqs,
        algebraic_vars=algebraic_vars,
        reformulated_vars=reformulated_vars,
        init_eqs={
            va_c: v_d_c * sym.cos(theta) + v_q_c * sym.sin(theta),
            vb_c: ((-vfactory.add_const(0.5) * v_d_c - (sqrt3 / vfactory.add_const(2)) * v_q_c) * sym.cos(theta)
                   + ((sqrt3 / vfactory.add_const(2)) * v_d_c - vfactory.add_const(0.5) * v_q_c) * sym.sin(theta)),
            vc_c: ((-vfactory.add_const(0.5) * v_d_c + (sqrt3 / vfactory.add_const(2)) * v_q_c) * sym.cos(theta)
                   + ((-sqrt3 / vfactory.add_const(2)) * v_d_c - vfactory.add_const(0.5) * v_q_c) * sym.sin(theta)),
        },
    )
    inv_park_block.add(trig_block)

    return inv_park_block, (va_c, vb_c, vc_c), aux_vars

def park_transform_block(vfactory: VarFactory, v_abc: list[Var], theta: Var, multilinear:bool = False, aux_vars = None, name:str = ''):
    """
    Create a symbolic Park transform (abc → dq) block for voltages and currents.

    i_abc = [i_a, i_b, i_c]
    v_abc = [v_a, v_b, v_c]
    theta : grid angle (from PLL)
    """

    v_a, v_b, v_c = v_abc

    # dq variables
    x_d = Var(name + '_d')
    x_q = Var(name + '_q')

    if not multilinear:
        algebraic_eqs = [
            # dq voltages
            x_d - (vfactory.add_const(1/3) * (
                (vfactory.add_const(2) * sym.cos(theta) * v_a)
                + (-sym.cos(theta) + vfactory.add_const(np.sqrt(3)) * sym.sin(theta)) * v_b
                + (-sym.cos(theta) - vfactory.add_const(np.sqrt(3)) * sym.sin(theta)) * v_c
            )),

            x_q - (vfactory.add_const(1/3) * (
                (vfactory.add_const(2) * sym.sin(theta) * v_a)
                + (-sym.sin(theta) - vfactory.add_const(np.sqrt(3)) * sym.cos(theta)) * v_b
                + (-sym.sin(theta) + vfactory.add_const(np.sqrt(3)) * sym.cos(theta)) * v_c
            )),
        ]
        algebraic_vars = [x_d, x_q]
        trig_block = Block()
        aux_vars = None
    else:
        trig_block, u_cos, u_sin = symbolic_ml.trig_transform(vfactory, theta, type="usual")
        sqrt3 = vfactory.add_const(np.sqrt(3.0))
        third = vfactory.add_const(1.0 / 3.0)
        algebraic_eqs = [
            x_d - third * (vfactory.add_const(2.0) * u_cos * v_a
                           + (-u_cos + sqrt3 * u_sin) * v_b
                           + (-u_cos - sqrt3 * u_sin) * v_c),
            x_q - third * (vfactory.add_const(2.0) * u_sin * v_a
                           + (-u_sin - sqrt3 * u_cos) * v_b
                           + (-u_sin + sqrt3 * u_cos) * v_c),
        ]
        algebraic_vars = [x_d, x_q]
        aux_vars = (u_cos, u_sin)


    park_block = Block(
        algebraic_eqs=algebraic_eqs,
        algebraic_vars=algebraic_vars
    )
    park_block.add(trig_block)

    return park_block, (x_d, x_q), aux_vars

def pll_transform(vfactory: VarFactory, v_abc, multilinear:bool = False, name:str = ''):
    """
    EMT PLL using instantaneous three-phase voltages.
    """
    theta = Var('theta')
    omega = Var('omega')
    omega_base = vfactory.add_var('omega_base')
    Kp_pll = vfactory.add_var('Kp_pll')      # proportional gain
    Ki_pll = vfactory.add_var('Ki_pll')      # integral gain

    park_theta = theta
    park_block, v_dq, aux_vars = park_transform_block(vfactory, v_abc, park_theta, multilinear=multilinear, name = name) 
    v_d, v_q = v_dq
    res_block = Block()
    pll_error = vfactory.add_var('u_PLL_pi')
    xi_pll = vfactory.add_var('xi_PLL')
    one = vfactory.add_const(1.0)
    d_theta = vfactory.add_diff_var(name='dt_1_theta', base_var=theta)
    d_xi_pll = vfactory.add_diff_var(name='dt_1_xi_PLL', base_var=xi_pll)
    integrator = Block(
        algebraic_eqs=[
            pll_error - v_d,
            omega - (one + Kp_pll * pll_error + Ki_pll * xi_pll),
        ],
        algebraic_vars=[pll_error, omega],
        state_eqs=[omega_base * omega, pll_error],
        state_vars=[theta, xi_pll],
        diff_vars=[d_theta, d_xi_pll],
        init_eqs={
            pll_error: v_d,
            xi_pll: vfactory.add_const(0.0),
            omega: one,
        },
        diff_init_eqs={d_theta: omega_base * omega, d_xi_pll: pll_error},
    )

    res_block.add(Block(event_dict={
        omega_base: vfactory.add_const(2.0 * math.pi * 50.0),
        Kp_pll: vfactory.add_const(0.03),
        Ki_pll: vfactory.add_const(0.2),
    }))
    res_block.add(integrator)
    res_block.add(park_block)
    return res_block, v_dq, omega, theta, aux_vars


def build_gfl_converter_model_emt(vfactory: VarFactory, inputs, 
                                  control1: ConverterControlType = ConverterControlType.Pac, 
                                  control2: ConverterControlType = ConverterControlType.Qac,
                                  multilinear:bool = False,
                                  frozen_voltage_source: bool = False):
    """
    Build power control loop model for Grid Following Converter for EMT simulation.
    Supports multiple control modes via ConverterControlType.

    Args:
        inputs: [vc_A, vc_B, vc_C, vg_A, vg_B, vg_C, i_A, i_B, i_C, Vdc,
                 Pt_vsc, Qt_vsc, Vpk_ref, phi_v_ref]
        control1: First control mode (typically active power or DC voltage related)
        control2: Second control mode (typically reactive power or AC voltage related)
        multilinear: Use multilinearization for Park transforms

    Returns:
        gfl_block: The complete converter block
        i_a_inj, i_b_inj, i_c_inj: Three-phase bus injection currents
        P, Q: Active and reactive power measurements
    """
    vc_a   = inputs[0]
    vc_b   = inputs[1]
    vc_c   = inputs[2]
    vg_a   = inputs[3]
    vg_b   = inputs[4]
    vg_c   = inputs[5]
    i_a_line = inputs[6]
    i_b_line = inputs[7]
    i_c_line = inputs[8]
    v_dc   = inputs[9]
    Pt_vsc = inputs[10]  # Power flow initial values
    Qt_vsc = inputs[11]
    Vpk_ref = inputs[12]
    phi_v_ref = inputs[13]

    algebraic_eqs = []
    algebraic_vars = []
    gfl_block = Block()
    control_blocks = []
    
    # ==============================
    # Inputs and variables
    # ==============================

    vc_abc = [vc_a, vc_b, vc_c]
    vg_abc = [vg_a, vg_b, vg_c]
    i_abc = [i_a_line, i_b_line, i_c_line]

    pll_block, v_dq, omega, theta, aux_vars = pll_transform(vfactory, vg_abc, multilinear=multilinear, name = 'vg')
    v_d_g = v_dq[0]
    v_q_g = v_dq[1]
    park_theta = theta
    i_park_block, i_dq, _ = park_transform_block(vfactory, i_abc, park_theta, multilinear=multilinear, name='i_line')
    i_d_line = i_dq[0]
    i_q_line = i_dq[1]
    vc_park_block, vc_dq, _ = park_transform_block(vfactory, vc_abc, park_theta, multilinear=multilinear, name='vc')
    vc_d = vc_dq[0]
    vc_q = vc_dq[1]

    # Parameters
    Kp_icl = vfactory.add_var('Kp_icl')     # proportional gain for inner current loop
    Ki_icl = vfactory.add_var('Ki_icl')     # integral gain for inner current loop
    Kp_pol = vfactory.add_var('Kp_pol')     # proportional gain for outer power loop
    Ki_pol = vfactory.add_var('Ki_pol')     # integral gain for outer power loop
    Kp_vac = vfactory.add_var('Kp_vac')     # proportional gain for AC voltage control
    Ki_vac = vfactory.add_var('Ki_vac')     # integral gain for AC voltage control
    L = vfactory.add_var('L') 

    # Current measurements come directly from the external line current inputs.
    i_d = i_d_line
    i_q = i_q_line
    i_d_ref = vfactory.add_var('i_d_ref')
    i_q_ref = vfactory.add_var('i_q_ref')
    
    # POWER LOOP 
    P = vfactory.add_var('P')
    Q = vfactory.add_var('Q') 
    P_ref = vfactory.add_var('P_ref') 
    Q_ref = vfactory.add_var('Q_ref') 
    
    # Voltage references for control modes
    Vm_ac_ref = vfactory.add_var('Vm_ac_ref')
    
    v_d_c = vfactory.add_var('v_d_c')
    v_q_c = vfactory.add_var('v_q_c')
    v_d_c_ref = vfactory.add_var('v_d_c_ref')
    v_q_c_ref = vfactory.add_var('v_q_c_ref')

    event_dict = {
        Kp_icl: vfactory.add_const(1.0),
        Ki_icl: vfactory.add_const(20.0),
        Kp_pol: vfactory.add_const(0.5),
        Ki_pol: vfactory.add_const(10.0),
        Kp_vac: vfactory.add_const(0.1),
        Ki_vac: vfactory.add_const(2.0),
        L: vfactory.add_const(0.1),
        P_ref: vfactory.add_const(None),
        Q_ref: vfactory.add_const(None),
        Vm_ac_ref: vfactory.add_const(1.0),
    }
    if frozen_voltage_source:
        event_dict[v_d_c_ref] = vfactory.add_const(0.0)
        event_dict[v_q_c_ref] = vfactory.add_const(1.0)

    # P and Q at the grid-side point of common coupling.
    algebraic_eqs.append(P - vfactory.add_const(1/2)*(v_q_g*i_q + v_d_g*i_d))
    algebraic_eqs.append(Q - vfactory.add_const(1/2)*(v_d_g*i_q - v_q_g*i_d))
    algebraic_vars.append(P)
    algebraic_vars.append(Q)

    # ==============================
    # CONTROL 1: Active Power Axis (i_q_ref)
    # ==============================
    # Control1 maps to i_q_ref (q-axis current controls active power)
    
    if control1 == ConverterControlType.Pac:
        # Active power control at AC side
        control_block_1, _ = tf_to_block(vfactory,
            num=[Ki_pol, Kp_pol],
            den=[0, 1],
            x= P_ref - P,
            y = i_q_ref,
            name='Pac_ctrl'
        )
        control_blocks.append(control_block_1)
        
    elif control1 == ConverterControlType.Pdc:
        # Active power control at DC side (Pdc ~ Pac in steady state)
        # For simplicity, we control Pac which is approximately Pdc
        control_block_1, _ = tf_to_block(vfactory,
            num=[Ki_pol, Kp_pol],
            den=[0, 1],
            x= -(P_ref - P),  # P_ref represents desired Pdc
            y = i_q_ref,
            name='Pdc_ctrl'
        )
        control_blocks.append(control_block_1)

    else:
        raise ValueError(f"Control1 type {control1} not supported for GFL converter. "
                        f"Supported: Pac, Pdc")

    # ==============================
    # CONTROL 2: Reactive Power Axis (i_d_ref)
    # ==============================
    # Control2 maps to i_d_ref (d-axis current controls reactive power)
    
    if control2 == ConverterControlType.Qac or True:
        # Reactive power control
        control_block_2, _ = tf_to_block(vfactory,
            num=[Ki_pol, Kp_pol],
            den=[0, 1],
            x= Q - Q_ref,
            y = i_d_ref,
            name='Qac_ctrl'
        )
        control_blocks.append(control_block_2)
        

    # Physical current limit in instantaneous peak-current p.u.  Keep it as a
    # runtime parameter because a system-base model must scale the limit with
    # the converter rating; 1.2 is retained as the legacy default.
    I_max = vfactory.add_var('I_max')
    event_dict[I_max] = vfactory.add_const(1.2)
    operation = 'normal'
    limiter_blocks = []
    if operation == 'normal':
        if multilinear:
            # Smooth multilinear lift of the circular current limiter. Exact
            # complementarity is singular at i_q == i_q_ref, so a tiny squared-
            # current smoothing regularizes that switching surface. Runtime
            # equations remain linear/bilinear; roots occur only in init_eqs.
            eps_current_sq = vfactory.add_const(1e-5)
            limiter_smoothing = vfactory.add_const(1e-10)
            half = vfactory.add_const(0.5)

            iq_delta = vfactory.add_var("i_q_headroom_delta")
            iq_delta_aux = vfactory.add_var("i_q_headroom_delta_aux")
            iq_delta_sq = vfactory.add_var("i_q_headroom_delta_sq")
            iq_delta_block = Block(
                algebraic_eqs=[
                    iq_delta - (i_q - i_q_ref),
                    iq_delta_aux - iq_delta,
                    iq_delta_sq - iq_delta * iq_delta_aux - limiter_smoothing,
                ],
                algebraic_vars=[iq_delta, iq_delta_aux, iq_delta_sq],
                init_eqs={
                    iq_delta: i_q - i_q_ref,
                    iq_delta_aux: iq_delta,
                    iq_delta_sq: iq_delta * iq_delta_aux + limiter_smoothing,
                },
                name="gfl_iq_headroom_difference_lift",
            )
            iq_root_block, iq_delta_abs = symbolic_ml.ml_smooth_sqrt(
                vfactory, iq_delta_sq, name="gfl_iq_headroom"
            )
            iq_limit_axis = vfactory.add_var("i_q_limit_axis")
            iq_limit_axis_aux = vfactory.add_var("i_q_limit_axis_aux")
            iq_alias_block = Block(
                algebraic_eqs=[
                    iq_limit_axis - half * (i_q + i_q_ref + iq_delta_abs),
                    iq_limit_axis_aux - iq_limit_axis,
                ],
                algebraic_vars=[iq_limit_axis, iq_limit_axis_aux],
                init_eqs={
                    iq_limit_axis: half * (i_q + i_q_ref + sym.sqrt(iq_delta_sq)),
                    iq_limit_axis_aux: iq_limit_axis,
                },
                name="gfl_iq_limit_axis_lift",
            )

            id_headroom_raw = I_max * I_max - iq_limit_axis * iq_limit_axis_aux
            floor_delta = vfactory.add_var("i_d_headroom_floor_delta")
            floor_delta_aux = vfactory.add_var("i_d_headroom_floor_delta_aux")
            floor_delta_sq = vfactory.add_var("i_d_headroom_floor_delta_sq")
            floor_delta_block = Block(
                algebraic_eqs=[
                    floor_delta - (id_headroom_raw - eps_current_sq),
                    floor_delta_aux - floor_delta,
                    floor_delta_sq - floor_delta * floor_delta_aux - limiter_smoothing,
                ],
                algebraic_vars=[floor_delta, floor_delta_aux, floor_delta_sq],
                init_eqs={
                    floor_delta: id_headroom_raw - eps_current_sq,
                    floor_delta_aux: floor_delta,
                    floor_delta_sq: floor_delta * floor_delta_aux + limiter_smoothing,
                },
                name="gfl_id_headroom_floor_difference_lift",
            )
            floor_root_block, floor_delta_abs = symbolic_ml.ml_smooth_sqrt(
                vfactory, floor_delta_sq, name="gfl_id_headroom_floor"
            )
            id_headroom_sq = half * (
                id_headroom_raw + eps_current_sq + floor_delta_abs
            )
            sqrt_block, id_max = symbolic_ml.ml_smooth_sqrt(
                vfactory, id_headroom_sq, name="gfl_id_max"
            )
            id_sat_block, i_d_ref_sat = symbolic_ml.ml_smooth_hard_sat(
                vfactory, i_d_ref, -id_max, id_max,
                lam=1e-10, name="gfl_id_ref",
            )
            iq_sat_block, i_q_ref_sat = symbolic_ml.ml_smooth_hard_sat(
                vfactory, i_q_ref, -I_max, I_max,
                lam=1e-10, name="gfl_iq_ref",
            )
            limiter_blocks.extend([
                iq_delta_block,
                iq_root_block,
                iq_alias_block,
                floor_delta_block,
                floor_root_block,
                sqrt_block,
                id_sat_block,
                iq_sat_block,
            ])
        else:
            id_max = sym.sqrt(sym.max(I_max**2 - sym.max(i_q, i_q_ref)**2, vfactory.add_const(1e-5)))
            i_d_ref_sat = sym.hard_sat(i_d_ref, -id_max, id_max)
            i_q_ref_sat = sym.hard_sat(i_q_ref, -I_max, I_max)

    # Voltage Control Loop (Inner Current Loop)
    control_block_iq , vq_hat = tf_to_block(vfactory,
        num=[Ki_icl, Kp_icl],
        den=[0, 1],
        x= i_q - i_q_ref_sat,
        name='vq_hat'
    )
    control_block_id , vd_hat = tf_to_block(vfactory,
        num=[Ki_icl, Kp_icl],
        den=[0, 1],
        x= i_d - i_d_ref_sat,
        name='vd_hat'
    )

    if frozen_voltage_source:
        algebraic_eqs.append(v_d_c - v_d_c_ref)
        algebraic_eqs.append(v_q_c - v_q_c_ref)
    else:
        algebraic_eqs.append(v_d_c - (vd_hat + v_d_g - L*(omega)*i_q))
        algebraic_eqs.append(v_q_c - (vq_hat + v_q_g + L*(omega)*i_d))
    algebraic_eqs.append(vc_d - v_d_c)
    algebraic_eqs.append(vc_q - v_q_c)
    algebraic_eqs.append(vc_a + vc_b + vc_c)
    algebraic_vars.extend([v_d_c, v_q_c])

    sqrt3 = vfactory.add_const(np.sqrt(3.0))
    one_third = vfactory.add_const(1.0 / 3.0)
    two = vfactory.add_const(2.0)
    vc_d_init = one_third * (
        two * sym.cos(park_theta) * vc_a
        + (-sym.cos(park_theta) - sqrt3 * sym.sin(park_theta)) * vc_b
        + (-sym.cos(park_theta) + sqrt3 * sym.sin(park_theta)) * vc_c
    )
    vc_q_init = one_third * (
        two * sym.sin(park_theta) * vc_a
        + (-sym.sin(park_theta) + sqrt3 * sym.cos(park_theta)) * vc_b
        + (-sym.sin(park_theta) - sqrt3 * sym.cos(park_theta)) * vc_c
    )

    # Build initialization equations based on control modes. P and Q are seeded
    # from power-flow results via external mapping; making them depend on the
    # currents here creates a P <-> i_q and Q <-> i_d explicit-init cycle.
    init_eqs = {
        theta: phi_v_ref,
        omega: vfactory.add_const(1),
        v_d_g: vfactory.add_const(0),
        v_q_g: Vpk_ref,
        i_q: vfactory.add_const(2.0) * P / v_q_g,
        i_d: -vfactory.add_const(2.0) * Q / v_q_g,
        i_q_ref: i_q,
        i_d_ref: i_d,
        v_d_c: vc_d_init,
        v_q_c: vc_q_init,
        vd_hat: v_d_c - (v_d_g - L*(omega)*i_q),
        vq_hat: v_q_c - (v_q_g + L*(omega)*i_d),
    }
    
    # Add control-specific initialization
    if control1 in [ConverterControlType.Pac, ConverterControlType.Pdc]:
        init_eqs[P_ref] = P

    if control2 == ConverterControlType.Qac:
        init_eqs[Q_ref] = Q
    elif control2 == ConverterControlType.Vm_ac:
        init_eqs[Vm_ac_ref] = v_q_g
        # For AC voltage control, initialize i_d_ref based on initial reactive power
        init_eqs[i_d_ref] = -vfactory.add_const(2.0) * Q / v_q_g

    gfl_block_aux = Block(
        algebraic_eqs=algebraic_eqs,
        algebraic_vars=algebraic_vars,
        event_dict= event_dict,
        init_eqs=init_eqs,
        external_mapping={
            VarPowerFlowReferenceType.P:P,
            VarPowerFlowReferenceType.Q:Q,
        }
    )
    gfl_block.add(gfl_block_aux)

    # Add all control blocks
    for ctrl_block in control_blocks:
        gfl_block.add(ctrl_block)
    for limiter_block in limiter_blocks:
        gfl_block.add(limiter_block)
    gfl_block.add(control_block_id)
    gfl_block.add(control_block_iq)

    gfl_block.add(pll_block)
    gfl_block.add(i_park_block)
    gfl_block.add(vc_park_block)
    gfl_block.unify_blocks()

    return gfl_block, P, Q

def VscGflEmtBuild(vfactory: VarFactory, name: str = "",
                   control1: ConverterControlType = ConverterControlType.Pac,
                   control2: ConverterControlType = ConverterControlType.Qac,
                   frozen_voltage_source: bool = False,
                   multilinear: bool = False) -> EmtModelTemplate:
    """
    VSC GFL (Grid Following) EMT model
    with from side the DC bus and to side the AC bus
    
    Args:
        name: Model name
        control1: First control mode (Pac or Pdc)
        control2: Second control mode (Qac or Vm_ac)
    
    Supported control combinations:
        - Pac + Qac: Active and reactive power control
        - Pac + Vm_ac: Active power and AC voltage control
        - Pdc + Qac: DC power and reactive power control
        - Pdc + Vm_ac: DC power and AC voltage control
    """
    templ = EmtModelTemplate()
    templ.tpe = DeviceType.VscDevice
    # Inputs: converter-side voltage, grid-side voltage, measured line current, and DC voltage.
    inputs = [
        vfactory.add_var("vc_A", reference=VarPowerFlowReferenceType.v_A),
        vfactory.add_var("vc_B", reference=VarPowerFlowReferenceType.v_B),
        vfactory.add_var("vc_C", reference=VarPowerFlowReferenceType.v_C),
        vfactory.add_var("vg_A"),
        vfactory.add_var("vg_B"),
        vfactory.add_var("vg_C"),
        vfactory.add_var("i_line_A"),
        vfactory.add_var("i_line_B"),
        vfactory.add_var("i_line_C"),
        vfactory.add_var("Vdc_", reference=VarPowerFlowReferenceType.Vdc),
    ]

    # Vars:
    Pt_vsc = vfactory.add_var('Pt_vsc')
    Qt_vsc = vfactory.add_var('Qt_vsc')
    Vpk_ref = vfactory.add_var('Vpk_ref')
    phi_v_ref = vfactory.add_var('phi_v_ref')
    i_a_t = vfactory.add_var('i_a_f')
    i_b_t = vfactory.add_var('i_b_f')
    i_c_t = vfactory.add_var('i_c_f')
    i_dc = vfactory.add_var('i_dc')
    P_conv = vfactory.add_var('P_conv')
    i_conv_dc = vfactory.add_var('i_conv_dc') if multilinear else None
    v_dc_cap = vfactory.add_var('Vdc_cap')
    d_v_dc_cap = vfactory.add_diff_var(name='dt_1_Vdc_cap', base_var=v_dc_cap)

    # Parameters:
    bt = vfactory.add_var('bt')
    gt = vfactory.add_var('gt')
    Qf = vfactory.add_var('Qf')
    a0 = vfactory.add_var('a0')
    a1 = vfactory.add_var('a1')
    a2 = vfactory.add_var('a2')
    Cdc = vfactory.add_var('Cdc')
    
    # Build the converter model with specified control modes
    gfl_block, P, Q = build_gfl_converter_model_emt(
        vfactory=vfactory,
        inputs=[*inputs, Pt_vsc, Qt_vsc, Vpk_ref, phi_v_ref],
        control1=control1,
        control2=control2,
        multilinear=multilinear,
        frozen_voltage_source=frozen_voltage_source
    )

    event_dict = {
        Qf: vfactory.add_const(0.0),
        bt: vfactory.add_const(0.0),
        gt: vfactory.add_const(0.1),
        a0: vfactory.add_const(0.0),
        a1: vfactory.add_const(0.0),
        a2: vfactory.add_const(0.0),
        Cdc: vfactory.add_const(10.0),
        Vpk_ref: vfactory.add_const(None),
        phi_v_ref: vfactory.add_const(None),
    }
    
    # EMT model outputs three-phase currents
    p_conv_init = -(inputs[0] * i_a_t + inputs[1] * i_b_t + inputs[2] * i_c_t) / vfactory.add_const(3.0)
    eps_vdc = vfactory.add_const(1.0e-10)
    algebraic_eqs = [
            Pt_vsc + P,
            Qt_vsc + Q,
            P_conv + (inputs[0] * i_a_t + inputs[1] * i_b_t + inputs[2] * i_c_t) / vfactory.add_const(3.0),
            v_dc_cap - inputs[9],
        ]
    algebraic_vars = [Pt_vsc, Qt_vsc, i_a_t, i_b_t, i_c_t, i_dc, P_conv]
    init_eqs = {
        v_dc_cap: inputs[9],
        P_conv: p_conv_init,
        i_dc: p_conv_init / inputs[9],
    }
    if multilinear:
        # Exact lifting of i_conv_dc = P_conv / Vdc.  Keeping the quotient in
        # the differential equation prevents construction of exact Phi/S
        # matrices even though the residual/Jacobian evaluator can simulate it.
        algebraic_eqs.append(i_conv_dc * (v_dc_cap + eps_vdc) - P_conv)
        algebraic_vars.append(i_conv_dc)
        init_eqs[i_conv_dc] = p_conv_init / inputs[9]
        dc_cap_rhs = (i_dc - i_conv_dc) / Cdc
    else:
        dc_cap_rhs = (i_dc - P_conv / (v_dc_cap + eps_vdc)) / Cdc

    vsc_block = Block(
        algebraic_eqs=algebraic_eqs,
        algebraic_vars=algebraic_vars,
        state_eqs=[dc_cap_rhs],
        state_vars=[v_dc_cap],
        diff_vars=[d_v_dc_cap],
        event_dict= event_dict,
        init_eqs=init_eqs,
        diff_init_eqs={
            d_v_dc_cap: vfactory.add_const(0.0),
        },
        external_mapping={
            VarPowerFlowReferenceType.P: P,
            VarPowerFlowReferenceType.Q: Q,
        },
        in_vars= inputs,
        out_vars = []
    )
    vsc_block.external_mapping = {
        VarPowerFlowReferenceType.P: P,
        VarPowerFlowReferenceType.Q: Q,
        VarPowerFlowReferenceType.v_A: inputs[0],
        VarPowerFlowReferenceType.v_B: inputs[1],
        VarPowerFlowReferenceType.v_C: inputs[2],
        VarPowerFlowReferenceType.Vdc: inputs[9],
        VarPowerFlowReferenceType.Idc: i_dc,
        VarPowerFlowReferenceType.Pt: Pt_vsc,
        VarPowerFlowReferenceType.Qt: Qt_vsc,
        VarPowerFlowReferenceType.i_A: i_a_t,
        VarPowerFlowReferenceType.i_B: i_b_t,
        VarPowerFlowReferenceType.i_C: i_c_t,
        VarPowerFlowReferenceType.Vpk: Vpk_ref,
        VarPowerFlowReferenceType.phi_v: phi_v_ref,
    }

    vsc_block.add(gfl_block)
    vsc_block.name = 'gfl_block_emt'

    templ.block = vsc_block
    return templ
