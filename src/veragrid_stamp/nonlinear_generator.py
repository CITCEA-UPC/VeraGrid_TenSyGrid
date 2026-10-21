"""STAMP winding-current synchronous generator, AC4A and IEEEG1."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .bases import RMS_LL_TO_PEAK_LN
from .parameters import OMEGA_BASE, SYSTEM_BASE_MVA, StampSynchronousGeneratorParameters
from .rotor_circuit import derive_rotor_circuit


def build_stamp_generator_rms(vf: Any, p: StampSynchronousGeneratorParameters, name: str) -> Any:
    from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
    from VeraGridEngine.Utils.Symbolic.block import Block
    from VeraGridEngine.Utils.Symbolic import symbolic as sym
    from VeraGridEngine.enumerations import DeviceType, VarPowerFlowReferenceType

    c = vf.add_const
    rotor = derive_rotor_circuit(p)
    vm = vf.add_var(f"{name}.Vm", reference=VarPowerFlowReferenceType.Vm)
    va = vf.add_var(f"{name}.Va", reference=VarPowerFlowReferenceType.Va)
    pg = vf.add_var(f"{name}.P", reference=VarPowerFlowReferenceType.P)
    qg = vf.add_var(f"{name}.Q", reference=VarPowerFlowReferenceType.Q)
    rotor_angle = vf.add_var(f"{name}.rotor_angle")

    igq, igd = vf.add_var(f"{name}.ig_q"), vf.add_var(f"{name}.ig_d")
    isq, isd = vf.add_var(f"{name}.is_q"), vf.add_var(f"{name}.is_d")
    ifd, ikd = vf.add_var(f"{name}.if_d"), vf.add_var(f"{name}.ik_d")
    ik1q, ik2q = vf.add_var(f"{name}.ik1_q"), vf.add_var(f"{name}.ik2_q")
    omega = vf.add_var(f"{name}.w_pu")

    sl2g = p.rated_mva / SYSTEM_BASE_MVA
    # STAMP uses distinct peak phase-neutral machine bases, not Sl2g for
    # voltage/current/impedance conversions.
    system_kv = 230.0
    vl2g = p.terminal_kv * RMS_LL_TO_PEAK_LN / system_kv
    il2g = ((2.0 / 3.0) * p.rated_mva / (p.terminal_kv * RMS_LL_TO_PEAK_LN)
            / (SYSTEM_BASE_MVA / system_kv))
    zl2g = (p.terminal_kv**2 / p.rated_mva) / (system_kv**2 / SYSTEM_BASE_MVA)
    rsnb = 300.0
    k_v = c(RMS_LL_TO_PEAK_LN)
    vgq = k_v * vm * sym.cos(va)
    vgd = -k_v * vm * sym.sin(va)
    ca, sa = sym.cos(rotor_angle), sym.sin(rotor_angle)
    # The six-winding machine currents live in the fixed rotor frame.  STAMP's
    # Il2g block scales *and rotates* them into the network frame before the
    # snubber-current balance is formed.
    isq_g = ca * isq + sa * isd
    isd_g = -sa * isq + ca * isd
    vsgq_g = c(rsnb * zl2g) * (c(il2g) * isq_g - igq)
    vsgd_g = c(rsnb * zl2g) * (c(il2g) * isd_g - igd)
    vsgq = (ca * vsgq_g - sa * vsgd_g) / c(vl2g)
    vsgd = (sa * vsgq_g + ca * vsgd_g) / c(vl2g)

    ll, lmd, lmq = rotor.ll, rotor.lmd, rotor.lmq
    m = np.asarray([
        [-ll-lmq, 0, 0, 0, lmq, lmq],
        [0, -ll-lmd, lmd, lmd, 0, 0],
        [0, -lmd, rotor.lfd+lmd, lmd, 0, 0],
        [0, -lmd, lmd, rotor.l1d+lmd, 0, 0],
        [-lmq, 0, 0, 0, rotor.l1q+lmq, lmq],
        [-lmq, 0, 0, 0, lmq, rotor.l2q+lmq],
    ]) / OMEGA_BASE
    minv = np.linalg.inv(m)
    currents = [isq, isd, ifd, ikd, ik1q, ik2q]
    resistive = [-p.rs * isq, -p.rs * isd, c(rotor.rf) * ifd, c(rotor.r1d) * ikd,
                 c(rotor.r1q) * ik1q, c(rotor.r2q) * ik2q]
    nq = omega * (c(-(ll+lmd)) * isd + c(lmd) * ifd + c(lmd) * ikd)
    nd = omega * (c(ll+lmq) * isq - c(lmq) * ik1q - c(lmq) * ik2q)

    # AC4A states and output field-winding voltage.
    vfilt, xlead, efd = vf.add_var(f"{name}.exc_filtx"), vf.add_var(f"{name}.exc_x1"), vf.add_var(f"{name}.exc_x2")
    vmag0 = vf.add_var(f"{name}.Vsg_mag0")
    vf0 = vf.add_var(f"{name}.vf_d0")
    vsg_mag = sym.sqrt(vsgq*vsgq+vsgd*vsgd+c(1e-12))
    d_vsg_mag = vsg_mag-vmag0
    vc = vfilt/c(p.exciter_tr)
    exc_a1 = (p.exciter_ta+p.exciter_tb)/(p.exciter_ta*p.exciter_tb)
    exc_a0 = 1.0/(p.exciter_ta*p.exciter_tb)
    exc_gain = -p.exciter_ka*rotor.rf/lmd
    exc_c1 = exc_gain*p.exciter_tc/(p.exciter_ta*p.exciter_tb)
    exc_c2 = exc_gain/(p.exciter_ta*p.exciter_tb)
    # Exact named form of MATLAB tf2ss(-KA*(TC*s+1)/((TB*s+1)
    # *(TA*s+1))*Rf/Lmd).  xlead/efd retain STAMP's exported state names.
    vf_winding = vf0+c(exc_c1)*xlead+c(exc_c2)*efd

    voltage_vector = [vsgq, vsgd, vf_winding, c(0), c(0), c(0)]
    rhs = [voltage_vector[i] - resistive[i] - ([nq, nd, c(0), c(0), c(0), c(0)][i]) for i in range(6)]
    winding_eqs = [sum((c(minv[row, col]) * rhs[col] for col in range(6)), c(0)) for row in range(6)]
    psi_q = c(-(ll+lmq))*isq + c(lmq)*(ik1q+ik2q)
    psi_d = c(-(ll+lmd))*isd + c(lmd)*(ifd+ikd)
    te = psi_d * isq - psi_q * isd

    # IEEEG1 governor and turbine.
    xgov, xt4, xt5, xt6, xt7 = [vf.add_var(f"{name}.{n}") for n in ("gov_x1","turbx1","turbx2","turbx3","turbx4")]
    pref = vf.add_var(f"{name}.Pref")
    # Exact named canonical realization of STAMP's MATLAB tf2ss blocks.  The
    # states are not physical cascade outputs: tf2ss returns controllable
    # companion coordinates.  Derive the coefficients from the IEEEG1 time
    # constants and steam fractions so parameter changes remain transparent.
    lag_polynomials = [np.asarray([time_constant, 1.0]) for time_constant in
                       (p.turbine_t4, p.turbine_t5, p.turbine_t6, p.turbine_t7)]
    turbine_den = np.asarray([1.0])
    for polynomial in lag_polynomials:
        turbine_den = np.polymul(turbine_den, polynomial)
    turbine_num = np.zeros_like(turbine_den)
    stage_gains = (p.turbine_k1+p.turbine_k2, p.turbine_k3+p.turbine_k4,
                   p.turbine_k5+p.turbine_k6, p.turbine_k7+p.turbine_k8)
    for stage, gain in enumerate(stage_gains):
        remaining = np.asarray([1.0])
        for polynomial in lag_polynomials[stage+1:]:
            remaining = np.polymul(remaining, polynomial)
        turbine_num[-remaining.size:] += gain*remaining
    turbine_num = turbine_num/turbine_den[0]
    turbine_den = turbine_den/turbine_den[0]
    direct_gain = turbine_num[0]
    turbine_c = turbine_num[1:]-direct_gain*turbine_den[1:]
    valve = xgov/c(p.governor_t3)
    tm = (sum((c(coefficient)*state for coefficient, state in
               zip(turbine_c, (xt4, xt5, xt6, xt7))), c(0))
          + c(direct_gain)*valve)
    mech_eq = (tm / omega - te - c(p.damping_d)*(omega-c(1))) / c(2*p.inertia_h)

    states = [igq, igd, isq, isd, ifd, ikd, ik1q, ik2q, omega, vfilt, xlead, efd, xgov, xt4, xt5, xt6, xt7]
    state_eqs = [
        c(OMEGA_BASE/(p.transformer_x*zl2g))*(vsgq_g-vgq-c(p.transformer_r*zl2g)*igq-c(p.transformer_x*zl2g)*igd),
        c(OMEGA_BASE/(p.transformer_x*zl2g))*(vsgd_g-vgd-c(p.transformer_r*zl2g)*igd+c(p.transformer_x*zl2g)*igq),
        *winding_eqs, mech_eq,
        -vfilt/c(p.exciter_tr)+d_vsg_mag,
        -c(exc_a1)*xlead-c(exc_a0)*efd+vc,
        xlead,
        -xgov/c(p.governor_t3)+pref-(omega-c(1))/c(p.governor_r),
        -sum((c(turbine_den[index+1])*state for index, state in
              enumerate((xt4, xt5, xt6, xt7))), c(0))+valve,
        xt4, xt5, xt6,
    ]
    power_p = c(1.5)*(vgq*igq+vgd*igd)
    power_q = c(1.5)*(vgq*igd-vgd*igq)

    # Analytical steady state. First establish transformer/internal voltage in
    # the network frame, then choose the fixed slack rotor frame so vd equation is zero.
    igq0 = (pg*vgq-qg*vgd)/(c(1.5)*(vgq*vgq+vgd*vgd)+c(1e-12))
    igd0 = (pg*vgd+qg*vgq)/(c(1.5)*(vgq*vgq+vgd*vgd)+c(1e-12))
    uq0 = vgq+c(p.transformer_r*zl2g)*igq0+c(p.transformer_x*zl2g)*igd0
    ud0 = vgd+c(p.transformer_r*zl2g)*igd0-c(p.transformer_x*zl2g)*igq0
    isgq0 = (igq0+uq0/c(rsnb*zl2g))/c(il2g)
    isgd0 = (igd0+ud0/c(rsnb*zl2g))/c(il2g)
    uq_l0, ud_l0 = uq0/c(vl2g), ud0/c(vl2g)
    eq0 = uq_l0+c(p.rs)*isgq0+c(p.xq)*isgd0
    ed0 = ud_l0+c(p.rs)*isgd0-c(p.xq)*isgq0
    # STAMP uses atan(Ed/Eq).  Avoid VeraGrid's symbolic atan2 here because
    # its interpreted initializer and generated-code paths use opposite
    # argument orders.
    angle0 = -sym.atan(ed0 / eq0)
    isq0 = ca*isgq0-sa*isgd0
    isd0 = sa*isgq0+ca*isgd0
    ifd0 = (vsgq+c(p.rs)*isq+c(p.xd)*isd)/c(lmd)
    init = {igq: igq0, igd: igd0, isq: isq0, isd: isd0,
            ifd: ifd0, ikd:c(0), ik1q:c(0), ik2q:c(0), omega:c(1),
            vfilt:c(0), xlead:c(0), efd:c(0),
            xgov:c(p.governor_t3)*te, xt4:c(0), xt5:c(0), xt6:c(0),
            xt7:te/c(turbine_den[-1])}
    events = {rotor_angle: angle0, vmag0:vsg_mag,
              vf0:c(rotor.rf)*ifd, pref:te}
    block = Block(state_vars=states, state_eqs=state_eqs, algebraic_vars=[pg,qg],
                  algebraic_eqs=[pg-power_p,qg-power_q], init_eqs=init, event_dict=events,
                  in_vars=[vm,va], out_vars=[pg,qg],
                  external_mapping={VarPowerFlowReferenceType.Vm:vm,VarPowerFlowReferenceType.Va:va,
                                    VarPowerFlowReferenceType.P:pg,VarPowerFlowReferenceType.Q:qg}, name=name)
    template=RmsModelTemplate(name=name); template.tpe=DeviceType.GeneratorDevice; template.block=block
    return template
