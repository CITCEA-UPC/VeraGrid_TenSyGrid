"""Nonlinear STAMP converter RMS models for VeraGrid.

The electrical equations follow STAMP's transformer and damped LCL topology.
Controllers are expressed as differential/algebraic equations so VeraGrid can
initialize them from its own solved terminal P/Q/V operating point.
"""

from __future__ import annotations

from typing import Any

from .bases import RMS_LL_TO_PEAK_LN
from .parameters import OMEGA_BASE, StampConverterParameters


def build_stamp_converter_rms(vf: Any, params: StampConverterParameters, name: str,
                              reference_omega: Any | None = None) -> Any:
    from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
    from VeraGridEngine.Utils.Symbolic.block import Block
    from VeraGridEngine.Utils.Symbolic import symbolic as sym
    from VeraGridEngine.enumerations import DeviceType, VarPowerFlowReferenceType

    c = vf.add_const
    vm = vf.add_var(f"{name}.Vm", reference=VarPowerFlowReferenceType.Vm)
    va = vf.add_var(f"{name}.Va", reference=VarPowerFlowReferenceType.Va)
    p = vf.add_var(f"{name}.P", reference=VarPowerFlowReferenceType.P)
    q = vf.add_var(f"{name}.Q", reference=VarPowerFlowReferenceType.Q)

    theta = vf.add_var(f"{name}.theta")
    igq, igd = vf.add_var(f"{name}.ig_q"), vf.add_var(f"{name}.ig_d")
    isq, isd = vf.add_var(f"{name}.is_q"), vf.add_var(f"{name}.is_d")
    ucq, ucd = vf.add_var(f"{name}.ucap_q"), vf.add_var(f"{name}.ucap_d")
    xi_iq, xi_id = vf.add_var(f"{name}.Ke_is_q"), vf.add_var(f"{name}.Ke_is_d")

    rtr, xtr = c(params.transformer_r), c(params.transformer_x)
    rc, xc = c(params.converter_r), c(params.converter_x)
    rac, bc = c(params.damping_r), c(params.capacitor_b)
    wb = c(OMEGA_BASE)
    frame_omega = c(1.0) if reference_omega is None else reference_omega
    eps = c(1e-12)

    voltage_l2g = c(RMS_LL_TO_PEAK_LN)
    ct, st = sym.cos(theta), sym.sin(theta)
    vgq = voltage_l2g * vm * sym.cos(va)
    vgd = -voltage_l2g * vm * sym.sin(va)
    # STAMP's Rac is in series with the shunt capacitor.  The capacitor-bus
    # voltage is therefore u = ucap + Rac * (is - ig).
    uq = ucq + rac * (isq - igq)
    ud = ucd + rac * (isd - igd)
    measured_p = c(1.5) * (vgq * igq + vgd * igd)
    measured_q = c(1.5) * (vgq * igd - vgd * igq)

    # STAMP measures both filter currents and the capacitor-bus voltage through
    # identical first-order delays before feeding the controllers.
    isq_m, isd_m = vf.add_var(f"{name}.isq_md"), vf.add_var(f"{name}.isd_md")
    igq_m, igd_m = vf.add_var(f"{name}.igq_md"), vf.add_var(f"{name}.igd_md")
    uq_m, ud_m = vf.add_var(f"{name}.uq_md"), vf.add_var(f"{name}.ud_md")
    measurement_states = [isq_m, isd_m, igq_m, igd_m, uq_m, ud_m]
    def local(q_global: Any, d_global: Any) -> tuple[Any, Any]:
        return ct * q_global - st * d_global, st * q_global + ct * d_global

    isq_local, isd_local = local(isq, isd)
    igq_local, igd_local = local(igq, igd)
    uq_local, ud_local = local(uq, ud)
    measurement_inputs = [isq_local, isd_local, igq_local, igd_local, uq_local, ud_local]
    measurement_eqs = [(source - delayed) / c(params.measurement_delay)
                       for delayed, source in zip(measurement_states, measurement_inputs)]

    event = {}
    init = {}

    vgq_global = voltage_l2g * vm * sym.cos(va)
    vgd_global = -voltage_l2g * vm * sym.sin(va)
    v2_global = vgq_global * vgq_global + vgd_global * vgd_global + eps
    igq_global = (p * vgq_global - q * vgd_global) / (c(1.5) * v2_global)
    igd_global = (p * vgd_global + q * vgq_global) / (c(1.5) * v2_global)
    uq_global = vgq_global + rtr * igq_global + xtr * igd_global
    ud_global = vgd_global + rtr * igd_global - xtr * igq_global
    init[theta] = -sym.atan2(ud_global, uq_global)
    init[igq] = igq_global
    init[igd] = igd_global
    uq0 = vgq + rtr * igq + xtr * igd
    ud0 = vgd + rtr * igd - xtr * igq
    cap_a = rac * bc
    init[ucq] = (uq0 - cap_a * ud0) / (c(1.0) + cap_a * cap_a)
    init[ucd] = (cap_a * uq0 + ud0) / (c(1.0) + cap_a * cap_a)
    init[isq] = igq + bc * ucd
    init[isd] = igd - bc * ucq
    init.update({delayed: source for delayed, source in zip(measurement_states, measurement_inputs)})

    if params.mode == "GFOR":
        pf, qf = vf.add_var(f"{name}.p_filt"), vf.add_var(f"{name}.q_filt")
        xi_vq, xi_vd = vf.add_var(f"{name}.Ke_u_q"), vf.add_var(f"{name}.Ke_u_d")
        igq_ff, igd_ff = vf.add_var(f"{name}.igq_ff"), vf.add_var(f"{name}.igd_ff")
        pref, qref, vref = vf.add_var(f"{name}.P_ref"), vf.add_var(f"{name}.Q_ref"), vf.add_var(f"{name}.V_ref")
        measured_local_p = c(1.5) * (uq_m * igq_m + ud_m * igd_m)
        measured_local_q = c(1.5) * (uq_m * igd_m - ud_m * igq_m)
        event.update({pref: measured_local_p, qref: measured_local_q, vref: uq_local})
        omega = c(1.0) + c(params.frequency_droop_gain) * (pref - pf)
        vq_ref = vref + c(params.voltage_droop_gain) * (qref - qf)
        vd_ref = c(0.0)
        kp_v, ki_v = params.voltage_pi_gains
        isq_ref = c(kp_v) * (vq_ref - uq_m) + c(ki_v) * xi_vq + bc * ud_m + igq_ff
        isd_ref = c(kp_v) * (vd_ref - ud_m) + c(ki_v) * xi_vd - bc * uq_m + igd_ff
        state_vars = [theta, pf, qf, xi_vq, xi_vd, igq_ff, igd_ff]
        state_eqs = [wb * (omega - frame_omega), (measured_local_p - pf) / c(params.frequency_droop_tau),
                     (measured_local_q - qf) / c(params.voltage_droop_tau), vq_ref - uq_m, vd_ref - ud_m,
                     (igq_m - igq_ff) / c(params.current_feedforward_tau),
                     (igd_m - igd_ff) / c(params.current_feedforward_tau)]
        init.update({pf: measured_local_p, qf: measured_local_q,
                     xi_vq: (isq_local - bc * ud_local - igq_local) / c(ki_v),
                     xi_vd: (isd_local + c(kp_v) * ud_local + bc * uq_local - igd_local) / c(ki_v),
                     igq_ff: igq_local, igd_ff: igd_local})
    else:
        pll, xp, xq = vf.add_var(f"{name}.pll_x"), vf.add_var(f"{name}.Ke_P"), vf.add_var(f"{name}.Ke_Q")
        wf, qf = vf.add_var(f"{name}.w_filt"), vf.add_var(f"{name}.q_filt")
        pref, qref = vf.add_var(f"{name}.P_ref"), vf.add_var(f"{name}.Q_ref")
        umag_ref = vf.add_var(f"{name}.Umag_ref")
        kp_pll, ki_pll = params.pll_gains
        omega = c(1.0) - c(kp_pll) * ud_m - c(ki_pll) * pll
        p_ref_droop = pref - c(params.frequency_droop_gain) * wf
        local_umag = sym.sqrt(uq_m * uq_m + ud_m * ud_m + eps)
        voltage_error = umag_ref - local_umag
        q_ref_droop = qref + c(params.voltage_droop_gain) * qf
        kp_p = params.current_settling_time / params.active_power_tau * 230.0
        ki_p = 230.0 / params.active_power_tau
        local_p = c(1.5) * (uq_m * igq_m + ud_m * igd_m)
        local_q = c(1.5) * (uq_m * igd_m - ud_m * igq_m)
        event.update({pref: local_p, qref: local_q, umag_ref: local_umag})
        isq_ref = c(kp_p) * (p_ref_droop - local_p) + c(ki_p) * xp
        isd_ref = c(kp_p) * (q_ref_droop - local_q) + c(ki_p) * xq
        state_vars = [theta, pll, wf, qf, xp, xq]
        state_eqs = [wb * (omega - frame_omega), ud_m,
                     ((omega - c(1.0)) - wf) / c(params.frequency_droop_tau),
                     (voltage_error - qf) / c(params.voltage_droop_tau),
                     p_ref_droop - local_p, q_ref_droop - local_q]
        init.update({pll: c(0.0), wf: c(0.0), qf: voltage_error,
                     xp: (isq_local - c(kp_p) * (p_ref_droop - local_p)) / c(ki_p),
                     xq: (isd_local - c(kp_p) * (q_ref_droop - local_q)) / c(ki_p)})

    kp_i, ki_i = params.current_kp, params.current_ki
    vcq_local = c(kp_i) * (isq_ref - isq_m) + c(ki_i) * xi_iq + xc * isd_m + uq_m
    vcd_local = c(kp_i) * (isd_ref - isd_m) + c(ki_i) * xi_id - xc * isq_m + ud_m
    vcq = ct * vcq_local + st * vcd_local
    vcd = -st * vcq_local + ct * vcd_local
    state_vars.extend(measurement_states)
    state_eqs.extend(measurement_eqs)
    state_vars.extend([xi_iq, xi_id, igq, igd, isq, isd, ucq, ucd])
    state_eqs.extend([
        isq_ref - isq_m,
        isd_ref - isd_m,
        wb / xtr * (uq - vgq - rtr * igq - xtr * igd),
        wb / xtr * (ud - vgd - rtr * igd + xtr * igq),
        wb / xc * (vcq - ucq - (rc + rac) * isq + rac * igq - xc * isd),
        wb / xc * (vcd - ucd - (rc + rac) * isd + rac * igd + xc * isq),
        wb / bc * (isq - igq) - wb * ucd,
        wb / bc * (isd - igd) + wb * ucq,
    ])
    vcq0 = ucq + (rc + rac) * isq - rac * igq + xc * isd
    vcd0 = ucd + (rc + rac) * isd - rac * igd - xc * isq
    vcq0_local, vcd0_local = local(vcq0, vcd0)
    init[xi_iq] = (vcq0_local - xc * isd_local - uq_local) / c(ki_i)
    init[xi_id] = (vcd0_local + xc * isq_local - ud_local) / c(ki_i)

    block = Block(
        state_vars=state_vars,
        state_eqs=state_eqs,
        algebraic_vars=[p, q],
        algebraic_eqs=[p - measured_p, q - measured_q],
        event_dict=event,
        init_eqs=init,
        in_vars=[vm, va],
        out_vars=[p, q],
        external_mapping={VarPowerFlowReferenceType.Vm: vm, VarPowerFlowReferenceType.Va: va,
                          VarPowerFlowReferenceType.P: p, VarPowerFlowReferenceType.Q: q},
    )
    block.name = name
    template = RmsModelTemplate()
    template.tpe = DeviceType.StaticGeneratorDevice
    template.block = block
    return template
