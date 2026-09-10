"""Multilinear internal-filter composition for the EMT GFL converter."""

from __future__ import annotations

import numpy as np

from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
from VeraGridEngine.Utils.Symbolic.block import Block, Var, VarPowerFlowReferenceType, find_name_in_block
import VeraGridEngine.Utils.Symbolic.symbolic_ml as symbolic_ml


def _find(block: Block, name: str) -> Var | None:
    variable = find_name_in_block(name, block)
    if variable is not None:
        return variable
    variable = next((item for item in block.get_all_vars() if item.name == name), None)
    if variable is not None:
        return variable
    for owner in block.get_all_blocks():
        variable = next(
            (item for item in (*owner.event_dict.keys(), *owner.parameters.keys()) if item.name == name),
            None,
        )
        if variable is not None:
            return variable
    return None


def _inverse_park_trig_block(vf: VarFactory, theta: Var) -> tuple[Block, Var, Var]:
    """Represent cos(-theta), sin(-theta) with differential ML auxiliaries."""
    angle = vf.add_var("theta_aux_filter_inv")
    angle_block = Block(
        algebraic_vars=[angle],
        algebraic_eqs=[angle + theta],
        init_eqs={angle: -theta},
        name="filter_inverse_park_angle_aux",
    )
    trig, u_cos, u_sin = symbolic_ml.trig_transform(vf, angle, type="usual")
    trig.name = "filter_inverse_park_trig_transform"
    u_cos.name = "u_cos_filter_inv"
    u_sin.name = "u_sin_filter_inv"
    # Keep the trigonometric evolution equations differential. Moving them to
    # the algebraic set, as the standalone validation adapter did, makes the
    # assembled network Jacobian rectangular once the bus/device derivative
    # aliases are included.
    if angle.diff_var is not None and theta.diff_var is not None:
        d_cos = _find(trig, "d_u_cos")
        d_sin = _find(trig, "d_u_sin")
        raw_cos = _find(trig, "u_cos")
        raw_sin = _find(trig, "u_sin")
        raw_cos = u_cos if raw_cos is None else raw_cos
        raw_sin = u_sin if raw_sin is None else raw_sin
        if d_cos is not None and d_sin is not None:
            d_angle = -theta.diff_var
            trig.diff_init_eqs.update({
                angle.diff_var: d_angle,
                d_cos: -d_angle * raw_sin,
                d_sin: d_angle * raw_cos,
            })
    trig.add(angle_block)
    return trig, u_cos, u_sin


def add_gfl_internal_filter_multilinear(
    vf: VarFactory,
    frequency_hz: float,
    model: Block,
) -> None:
    """Attach the validated multilinear RL filter to a GFL EMT block.

    Current is positive from the grid toward the converter. The equations use
    the same peak-valued abc/dq and one-third three-phase power convention as
    :mod:`vsc_gfl_emt`.
    """
    theta = _find(model, "theta")
    vd_c, vq_c = _find(model, "v_d_c"), _find(model, "v_q_c")
    yvd, yvq = _find(model, "y_vd_hat"), _find(model, "y_vq_hat")
    vgd, vgq = _find(model, "vg_d"), _find(model, "vg_q")
    power_p, power_q = _find(model, "P"), _find(model, "Q")
    inductance = _find(model, "L")
    omega = _find(model, "omega")
    id_line, iq_line = _find(model, "i_line_d"), _find(model, "i_line_q")
    id_ref, iq_ref = _find(model, "i_d_ref"), _find(model, "i_q_ref")
    uvd, uvq = _find(model, "u_vd_hat"), _find(model, "u_vq_hat")
    bus_v = [_find(model, f"vg_{phase}") for phase in "ABC"]
    required_names = ["theta", "v_d_c", "v_q_c", "y_vd_hat", "y_vq_hat",
                      "vg_d", "vg_q", "P", "Q", "L", "omega", "i_line_d",
                      "i_line_q", "i_d_ref", "i_q_ref", "vg_A", "vg_B", "vg_C"]
    required = [theta, vd_c, vq_c, yvd, yvq, vgd, vgq, power_p, power_q,
                inductance, omega, id_line, iq_line, id_ref, iq_ref, *bus_v]
    missing = [name for name, item in zip(required_names, required) if item is None]
    if missing:
        raise RuntimeError("Incomplete GFL interface for multilinear internal filter: " + ", ".join(missing))

    resistance = vf.add_var("R_filter")
    converter_v = [vf.add_var(name) for name in ("va_v", "vb_v", "vc_v")]
    filter_i = [vf.add_var(f"i_filter_{phase}") for phase in "ABC"]
    filter_di = [vf.add_diff_var(f"dt_i_filter_{phase}", base_var=current)
                 for phase, current in zip("ABC", filter_i)]

    old_currents = [model.external_mapping[reference] for reference in (
        VarPowerFlowReferenceType.i_A,
        VarPowerFlowReferenceType.i_B,
        VarPowerFlowReferenceType.i_C,
    )]
    for old, new in zip(old_currents, filter_i):
        model.update_model(old, new)

    eliminated_uids = {current.uid for current in filter_i}
    for owner in model.get_all_blocks():
        equations = []
        for equation in owner.algebraic_eqs:
            text = str(equation)
            if uvd is not None and text.startswith("(u_vd_hat) - "):
                equations.append(uvd - (id_line - id_ref))
            elif uvq is not None and text.startswith("(u_vq_hat) - "):
                equations.append(uvq - (iq_line - iq_ref))
            elif text.startswith("(P) - ((0.5)") or text.startswith("(Q) - ((0.5)"):
                continue
            elif not (
                text.startswith("(v_d_c) - (((y_vd_hat) + (vg_d))")
                or text.startswith("(v_q_c) - (((y_vq_hat) + (vg_q))")
                or text in {"(vc_d) - (v_d_c)", "(vc_q) - (v_q_c)", "((vc_A) + (vc_B)) + (vc_C)"}
                or text.startswith("(vc_d) - ((0.3333333333333333)")
                or text.startswith("(vc_q) - ((0.3333333333333333)")
            ):
                equations.append(equation)
        owner.algebraic_eqs = equations
        owner.algebraic_vars = [
            variable for variable in owner.algebraic_vars
            if variable.name not in {"vc_d", "vc_q"} and variable.uid not in eliminated_uids
        ]
        owner.init_eqs = {
            variable: equation for variable, equation in owner.init_eqs.items()
            if variable.name not in {"vc_d", "vc_q"}
        }

    trig, cosine, sine = _inverse_park_trig_block(vf, theta)
    half = vf.add_const(0.5)
    sqrt3 = vf.add_const(np.sqrt(3.0))
    third = vf.add_const(1.0 / 3.0)
    omega_base = vf.add_const(2.0 * np.pi * frequency_hz)
    va, vb, vc = converter_v
    ia, ib, ic = filter_i
    vga, vgb, vgc = bus_v
    filter_block = Block(
        algebraic_vars=converter_v,
        state_vars=filter_i,
        diff_vars=filter_di,
        state_eqs=[
            omega_base * (vg - converter - resistance * current) / inductance
            for vg, converter, current in zip(bus_v, converter_v, filter_i)
        ],
        algebraic_eqs=[
            power_p - third * (vga * ia + vgb * ib + vgc * ic),
            power_q - (third / sqrt3) * ((vgb - vgc) * ia + (vgc - vga) * ib + (vga - vgb) * ic),
            vd_c - (yvd + vgd + resistance * id_line - inductance * omega * iq_line),
            vq_c - (yvq + vgq + resistance * iq_line + inductance * omega * id_line),
            va - (vd_c * cosine + vq_c * sine),
            vb - (vd_c * (-half * cosine - half * sqrt3 * sine)
                  + vq_c * (-half * sine + half * sqrt3 * cosine)),
            vc - (vd_c * (-half * cosine + half * sqrt3 * sine)
                  + vq_c * (-half * sine - half * sqrt3 * cosine)),
        ],
        event_dict={resistance: vf.add_const(0.0)},
        name="internal_vsc_filter_rl_multilinear",
    )
    filter_block.add(trig)
    model.add(filter_block)
    model.unify_blocks()
