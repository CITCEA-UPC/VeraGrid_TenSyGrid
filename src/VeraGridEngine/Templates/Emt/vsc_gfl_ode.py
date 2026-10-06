# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Experimental algebraic-free GFL converter model.

Kept separate from vsc_gfl_emt so dynamic-lifting and S/Phi experiments do not
alter or obscure the established block-based GFL EMT template.
"""

from dataclasses import dataclass

import numpy as np

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Utils.Symbolic.block import Block, Var
import VeraGridEngine.Utils.Symbolic.symbolic as sym



@dataclass(frozen=True)
class GflOdeObservables:
    """Computed signals of the state-only GFL model.

    These are symbolic expressions rather than solver unknowns.  Keeping them
    outside ``Block.algebraic_vars`` is what makes the model an explicit ODE.
    """

    p: sym.Expr
    q: sym.Expr
    vc_abc: tuple[sym.Expr, sym.Expr, sym.Expr]
    vg_dq: tuple[sym.Expr, sym.Expr]
    i_dq: tuple[sym.Expr, sym.Expr]
    omega: sym.Expr
    vdc: sym.Expr
    dc_energy: sym.Expr


def _park_expressions(vfactory: VarFactory, abc, cosine, sine):
    """Park transform as expressions, without introducing algebraic variables."""
    a, b, c = abc
    third = vfactory.add_const(1.0 / 3.0)
    two = vfactory.add_const(2.0)
    sqrt3 = vfactory.add_const(np.sqrt(3.0))
    d_axis = third * (
        two * cosine * a
        + (-cosine + sqrt3 * sine) * b
        + (-cosine - sqrt3 * sine) * c
    )
    q_axis = third * (
        two * sine * a
        + (-sine - sqrt3 * cosine) * b
        + (-sine + sqrt3 * cosine) * c
    )
    return d_axis, q_axis


def _inverse_park_expressions(vfactory: VarFactory, dq, cosine, sine):
    """Inverse Park transform as expressions, without algebraic variables."""
    d_axis, q_axis = dq
    half = vfactory.add_const(0.5)
    sqrt3half = vfactory.add_const(np.sqrt(3.0) / 2.0)
    return (
        d_axis * cosine + q_axis * sine,
        d_axis * (-half * cosine + sqrt3half * sine)
        + q_axis * (-half * sine - sqrt3half * cosine),
        d_axis * (-half * cosine - sqrt3half * sine)
        + q_axis * (-half * sine + sqrt3half * cosine),
    )

def _dynamic_circular_limiter(
        vfactory: VarFactory,
        i_d_command,
        i_q_command,
        current_limit,
        time_constant,
):
    """Smooth multi-affine current limiter with an invariant circular boundary.

    Three synchronized copies per axis avoid repeated factors.  From an initial
    point inside the circle, the common barrier factor tends to zero at I_max,
    so an excessive command approaches but does not cross the boundary in
    continuous time.
    """
    i_d_states = [vfactory.add_var(f"i_d_ref_sat_{role}") for role in ("out", "a", "b")]
    i_q_states = [vfactory.add_var(f"i_q_ref_sat_{role}") for role in ("out", "a", "b")]
    i_d_out, i_d_a, i_d_b = i_d_states
    i_q_out, i_q_a, i_q_b = i_q_states
    barrier = vfactory.add_const(1.0) - (
        i_d_a * i_d_b + i_q_a * i_q_b
    ) / (current_limit * current_limit)
    d_rhs = (i_d_command - i_d_out) * barrier / time_constant
    q_rhs = (i_q_command - i_q_out) * barrier / time_constant
    states = [*i_d_states, *i_q_states]
    equations = [d_rhs, d_rhs, d_rhs, q_rhs, q_rhs, q_rhs]
    init_eqs = {
        i_d_out: i_d_command,
        i_d_a: i_d_out,
        i_d_b: i_d_out,
        i_q_out: i_q_command,
        i_q_a: i_q_out,
        i_q_b: i_q_out,
    }
    return i_d_out, i_q_out, states, equations, init_eqs

def build_gfl_converter_model_ode(
        vfactory: VarFactory,
        inputs: list[Var] | None = None,
        *,
        multilinear: bool = False,
        enable_current_limiter: bool = False,
        dc_state: str = "energy",
) -> tuple[Block, GflOdeObservables]:
    """Build a standalone GFL converter containing no algebraic unknowns.

    By default, ``inputs`` are
    ``[vg_A, vg_B, vg_C, P_dc_source, P_ref, Q_ref]`` and capacitor energy is
    the DC state. With ``dc_state="voltage"``, the fourth input is instead
    ``i_dc_source`` and the original voltage/quotient dynamics are retained.
    The
    physical plant (filter currents and DC capacitor) is included, so converter
    phase voltage is computed internally and line current is a state/output.

    With ``multilinear=True`` three synchronized differential sine/cosine pairs
    are used by the voltage Park transform, current Park transform, and inverse
    Park transform.  The copies have identical dynamics and initial values, so
    they remain equal in continuous time while preventing repeated factors such
    as ``u_sin**2`` after symbolic expansion.  The remaining DC quotient and,
    when enabled, circular current limiter are handled separately.
    """
    if dc_state not in {"energy", "voltage"}:
        raise ValueError("dc_state must be 'energy' or 'voltage'")
    dc_source_name = "P_dc_source" if dc_state == "energy" else "i_dc_source"
    if inputs is None:
        inputs = [vfactory.add_var(name) for name in (
            "vg_A", "vg_B", "vg_C", dc_source_name, "P_ref", "Q_ref"
        )]
    if len(inputs) != 6:
        raise ValueError(
            "GFL ODE requires six inputs: vg_A/B/C, DC source, P_ref, Q_ref"
        )
    vg_abc = tuple(inputs[:3])
    dc_source, p_ref, q_ref = inputs[3:]

    # Genuine memory elements only.
    i_abc = tuple(vfactory.add_var(f"i_line_{phase}") for phase in "ABC")
    dc_state_var = vfactory.add_var("E_dc" if dc_state == "energy" else "Vdc_")
    xi_pll = vfactory.add_var("xi_PLL")
    xi_p = vfactory.add_var("xi_Pac_ctrl")
    xi_q = vfactory.add_var("xi_Qac_ctrl")
    xi_vd = vfactory.add_var("xi_vd_hat")
    xi_vq = vfactory.add_var("xi_vq_hat")

    state_vars = [*i_abc, dc_state_var, xi_pll, xi_p, xi_q, xi_vd, xi_vq]
    if multilinear:
        trig_pairs = {
            role: (
                vfactory.add_var(f"u_cos_{role}"),
                vfactory.add_var(f"u_sin_{role}"),
            )
            for role in ("v", "i", "o")
        }
        cosine_v, sine_v = trig_pairs["v"]
        cosine_i, sine_i = trig_pairs["i"]
        cosine_o, sine_o = trig_pairs["o"]
        trig_states = [item for role in ("v", "i", "o") for item in trig_pairs[role]]
        state_vars.extend(trig_states)
    else:
        theta = vfactory.add_var("theta")
        cosine_v = cosine_i = cosine_o = sym.cos(theta)
        sine_v = sine_i = sine_o = sym.sin(theta)
        state_vars.append(theta)

    # Runtime-tunable physical and controller parameters.
    parameter_defaults = {
        "omega_base": 2.0 * np.pi * 50.0,
        "R_filter": 0.01,
        "L": 0.10,
        "Cdc": 10.0,
        "Kp_pll": 0.01,
        "Ki_pll": 0.05,
        "Kp_pol": 0.05,
        "Ki_pol": 1.0,
        "Kp_icl": 0.05,
        "Ki_icl": 1.0,
        "I_max": 2.5,
        "limiter_time_constant": 5.0e-4,
    }
    parameters = {name: vfactory.add_var(name) for name in parameter_defaults}
    event_dict = {
        parameters[name]: vfactory.add_const(value)
        for name, value in parameter_defaults.items()
    }
    omega_base = parameters["omega_base"]
    resistance = parameters["R_filter"]
    inductance = parameters["L"]
    capacitance = parameters["Cdc"]
    vg_d, vg_q = _park_expressions(vfactory, vg_abc, cosine_v, sine_v)
    i_d, i_q = _park_expressions(vfactory, i_abc, cosine_i, sine_i)
    p = vfactory.add_const(0.5) * (vg_q * i_q + vg_d * i_d)
    q = vfactory.add_const(0.5) * (vg_d * i_q - vg_q * i_d)
    if dc_state == "energy":
        dc_energy = dc_state_var
        vdc = sym.sqrt(vfactory.add_const(2.0) * dc_energy / capacitance)
        dc_rhs = dc_source - p
    else:
        vdc = dc_state_var
        dc_energy = vfactory.add_const(0.5) * capacitance * vdc * vdc
        dc_rhs = (
            dc_source - p / (vdc + vfactory.add_const(1.0e-8))
        ) / capacitance
    pll_error = vg_d
    omega = (
        vfactory.add_const(1.0)
        + parameters["Kp_pll"] * pll_error
        + parameters["Ki_pll"] * xi_pll
    )

    p_error = p_ref - p
    q_error = q - q_ref
    i_q_ref = parameters["Kp_pol"] * p_error + xi_p
    i_d_ref = parameters["Kp_pol"] * q_error + xi_q
    limiter_states = []
    limiter_eqs = []
    limiter_init_eqs = {}
    if enable_current_limiter:
        if multilinear:
            i_d_ref_sat, i_q_ref_sat, limiter_states, limiter_eqs, limiter_init_eqs = (
                _dynamic_circular_limiter(
                    vfactory,
                    i_d_ref,
                    i_q_ref,
                    parameters["I_max"],
                    parameters["limiter_time_constant"],
                )
            )
            state_vars.extend(limiter_states)
        else:
            i_q_ref_sat = sym.hard_sat(
                i_q_ref, -parameters["I_max"], parameters["I_max"]
            )
            i_d_headroom = sym.sqrt(sym.max(
                parameters["I_max"] ** 2 - i_q_ref_sat ** 2,
                vfactory.add_const(1.0e-8),
            ))
            i_d_ref_sat = sym.hard_sat(i_d_ref, -i_d_headroom, i_d_headroom)
    else:
        i_q_ref_sat, i_d_ref_sat = i_q_ref, i_d_ref

    i_d_error = i_d_ref_sat - i_d
    i_q_error = i_q_ref_sat - i_q
    vd_hat = parameters["Kp_icl"] * i_d_error + xi_vd
    vq_hat = parameters["Kp_icl"] * i_q_error + xi_vq
    vc_d = vd_hat + vg_d + inductance * omega * i_q
    vc_q = vq_hat + vg_q - inductance * omega * i_d
    vc_abc = _inverse_park_expressions(vfactory, (vc_d, vc_q), cosine_o, sine_o)

    state_eqs = [
        *[
            omega_base * (vc - vg - resistance * current) / inductance
            for vc, vg, current in zip(vc_abc, vg_abc, i_abc)
        ],
        dc_rhs,
        pll_error,
        parameters["Ki_pol"] * p_error,
        parameters["Ki_pol"] * q_error,
        parameters["Ki_icl"] * i_d_error,
        parameters["Ki_icl"] * i_q_error,
    ]
    if multilinear:
        # All copies have exactly the same RHS.  The output-role pair is used as
        # the rotation multiplier because omega depends on the distinct
        # voltage-role pair; this avoids squared trigonometric states.
        for _role in ("v", "i", "o"):
            state_eqs.extend([
                -omega_base * omega * sine_o,
                omega_base * omega * cosine_o,
            ])
    else:
        state_eqs.append(omega_base * omega)
    state_eqs.extend(limiter_eqs)

    diff_vars = [
        vfactory.add_diff_var(f"dt_1_{state.name}", base_var=state)
        for state in state_vars
    ]

    zero = vfactory.add_const(0.0)
    init_eqs = {state: zero for state in state_vars}
    init_eqs[dc_state_var] = (
        vfactory.add_const(20.0) if dc_state == "energy"
        else vfactory.add_const(2.0)
    )
    if multilinear:
        for cosine_state, sine_state in trig_pairs.values():
            init_eqs[cosine_state] = vfactory.add_const(1.0)
            init_eqs[sine_state] = zero
    init_eqs.update(limiter_init_eqs)
    diff_init_eqs = {diff: zero for diff in diff_vars}

    block = Block(
        name="gfl_converter_ode",
        in_vars=list(inputs),
        out_vars=[*i_abc, dc_state_var],
        state_vars=state_vars,
        diff_vars=diff_vars,
        state_eqs=state_eqs,
        algebraic_vars=[],
        algebraic_eqs=[],
        event_dict=event_dict,
        init_eqs=init_eqs,
        diff_init_eqs=diff_init_eqs,
        reformulated_vars=(trig_states if multilinear else []),
    )
    return block, GflOdeObservables(
        p=p,
        q=q,
        vc_abc=vc_abc,
        vg_dq=(vg_d, vg_q),
        i_dq=(i_d, i_q),
        omega=omega,
        vdc=vdc,
        dc_energy=dc_energy,
    )
