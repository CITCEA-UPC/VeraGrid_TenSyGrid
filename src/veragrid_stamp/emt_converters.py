"""Three-phase EMT interface for the named STAMP GFOR/GFOL models.

The controller and LCL-filter states retain exactly the q-d equations used by
the validated RMS port.  Only the network interface changes: instantaneous
abc terminal voltages are Park-transformed into the synchronous network frame
and the q-d grid current is transformed back into three injected phase
currents.  Consequently the balanced fundamental-frequency envelope has the
same operating point and slow modes as the RMS model, while the surrounding
network is represented by VeraGrid's full three-phase EMT equations.
"""

from __future__ import annotations

import numpy as np

from .bases import RMS_LL_TO_PEAK_LN
from .parameters import OMEGA_BASE, StampConverterParameters


def build_stamp_converter_emt(vf, p: StampConverterParameters, name: str,
                              reference_omega=None):
    """Return a three-phase EMT model for one STAMP converter."""
    from VeraGridEngine.Devices.Dynamic.emt_template import EmtModelTemplate
    from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_pf_positive_sequence_init_refs
    from VeraGridEngine.Utils.Symbolic import symbolic as sym
    from VeraGridEngine.Utils.Symbolic.block import Block
    from VeraGridEngine.enumerations import DeviceType, ParamPowerFlowReferenceType, VarPowerFlowReferenceType

    c = vf.add_const
    z = c(0.0)
    one = c(1.0)
    shift = 2.0 * np.pi / 3.0

    va = vf.add_var(f"v_A_{name}", reference=VarPowerFlowReferenceType.v_A)
    vb = vf.add_var(f"v_B_{name}", reference=VarPowerFlowReferenceType.v_B)
    vc = vf.add_var(f"v_C_{name}", reference=VarPowerFlowReferenceType.v_C)
    dva = vf.add_var(f"d_v_A_{name}", reference=VarPowerFlowReferenceType.d_v_A)
    dvb = vf.add_var(f"d_v_B_{name}", reference=VarPowerFlowReferenceType.d_v_B)
    dvc = vf.add_var(f"d_v_C_{name}", reference=VarPowerFlowReferenceType.d_v_C)
    phase_power = [vf.add_var(f"{quantity}_{phase}_{name}", reference=reference)
                   for phase, quantity, reference in (
                       ("A", "P", VarPowerFlowReferenceType.P_A), ("A", "Q", VarPowerFlowReferenceType.Q_A),
                       ("B", "P", VarPowerFlowReferenceType.P_B), ("B", "Q", VarPowerFlowReferenceType.Q_B),
                       ("C", "P", VarPowerFlowReferenceType.P_C), ("C", "Q", VarPowerFlowReferenceType.Q_C))]
    _phi_v0, _phi_i0, _vpk0, _ipk0 = get_pf_positive_sequence_init_refs(
        v_a=va, v_b=vb, v_c=vc, d_v_a=dva, d_v_b=dvb, d_v_c=dvc,
        p_a=phase_power[0], q_a=phase_power[1], p_b=phase_power[2],
        q_b=phase_power[3], p_c=phase_power[4], q_c=phase_power[5],
        omega_base=c(OMEGA_BASE))

    theta_grid = vf.add_var(f"{name}.theta_grid")
    dtheta_grid = vf.add_diff_var(f"d_{name}.theta_grid", base_var=theta_grid)
    vq = (2.0 / 3.0) * (sym.sin(theta_grid) * va + sym.sin(theta_grid-shift) * vb + sym.sin(theta_grid+shift) * vc)
    vd = -(2.0 / 3.0) * (sym.cos(theta_grid) * va + sym.cos(theta_grid-shift) * vb + sym.cos(theta_grid+shift) * vc)

    # The PF references describe total three-phase injection.  These formulas
    # use STAMP's q=real, d=-imaginary convention and peak phase quantities.
    pref = vf.add_var(f"{name}.P_ref")
    qref = vf.add_var(f"{name}.Q_ref")
    eps = c(1e-12)
    # Freeze only the operating coefficients, not the live terminal signals.
    # This avoids embedding PF-only atan2 expressions in the runtime JIT and
    # uses the same exported STAMP operating point as the RMS audit.
    operating_points = {
        "GFOR": (1.02577, -0.961573414281591, -0.0437531356548132),
        "GFOL": (1.03235, -0.00230152645507726, -0.139271984262034),
    }
    vm_op, angle_deg_op, q_op = operating_points[p.mode]
    angle_op = np.deg2rad(angle_deg_op)
    # STAMP's converter equations are on a peak phase-to-neutral voltage base,
    # whereas the imported PF voltage is RMS line-to-line pu.  This is the
    # same conversion used by the validated RMS port.  Using sqrt(2) here
    # incorrectly treats the PF pu value as phase-neutral RMS and makes every
    # frozen controller/rotation operating coefficient too large by sqrt(3).
    peak_base = RMS_LL_TO_PEAK_LN
    vq0_value = peak_base * vm_op * np.cos(angle_op)
    vd0_value = -peak_base * vm_op * np.sin(angle_op)
    vq0 = c(vq0_value)
    vd0 = c(vd0_value)
    v20 = vq0*vq0 + vd0*vd0 + eps

    wb = c(OMEGA_BASE); rtr = c(p.transformer_r); ltr = c(p.transformer_x/OMEGA_BASE)
    rc = c(p.converter_r); lc = c(p.converter_x/OMEGA_BASE)
    cac = c(p.capacitor_b/OMEGA_BASE); rac = c(p.damping_r)
    igq0_value = (p.p_pu_system*vq0_value-q_op*vd0_value)/(1.5*(vq0_value**2+vd0_value**2+1e-12))
    igd0_value = (p.p_pu_system*vd0_value+q_op*vq0_value)/(1.5*(vq0_value**2+vd0_value**2+1e-12))
    uq0_value = vq0_value+p.transformer_r*igq0_value+p.transformer_x*igd0_value
    ud0_value = vd0_value+p.transformer_r*igd0_value-p.transformer_x*igq0_value
    cap_a_value = p.damping_r*p.capacitor_b
    ucapq0_value = (uq0_value-cap_a_value*ud0_value)/(1.0+cap_a_value**2)
    ucapd0_value = (cap_a_value*uq0_value+ud0_value)/(1.0+cap_a_value**2)
    isq0_value = igq0_value+p.capacitor_b*ucapd0_value
    isd0_value = igd0_value-p.capacitor_b*ucapq0_value
    igq0,igd0,uq0,ud0 = map(c,(igq0_value,igd0_value,uq0_value,ud0_value))
    ucapq0,ucapd0,isq0,isd0 = map(c,(ucapq0_value,ucapd0_value,isq0_value,isd0_value))
    # The operating point is close to the positive q axis, so the single-
    # argument form is quadrant-safe for this case and is supported by the EMT
    # structural JIT (which currently does not compile symbolic Func2/atan2).
    theta0_value = -np.arctan(ud0_value/uq0_value)
    ct = c(np.cos(theta0_value)); st = c(np.sin(theta0_value))

    # Named deviation coordinates are intentionally identical to the RMS port.
    isq_m=vf.add_var(f"{name}.isq_md"); isd_m=vf.add_var(f"{name}.isd_md")
    igq_m=vf.add_var(f"{name}.igq_md"); igd_m=vf.add_var(f"{name}.igd_md")
    uq_m=vf.add_var(f"{name}.uq_md"); ud_m=vf.add_var(f"{name}.ud_md")
    theta=vf.add_var(f"{name}.etheta_x")
    igq=vf.add_var(f"{name}.ig_q"); igd=vf.add_var(f"{name}.ig_d")
    isq=vf.add_var(f"{name}.is_q"); isd=vf.add_var(f"{name}.is_d")
    ucapq=vf.add_var(f"{name}.ucap_q"); ucapd=vf.add_var(f"{name}.ucap_d")
    xi_iq=vf.add_var(f"{name}.Ke_is_q"); xi_id=vf.add_var(f"{name}.Ke_is_d")
    # The EMT Park voltage is peak phase voltage on VeraGrid's conventional
    # phase base.  STAMP's q-d equations use peak phase-to-neutral voltage on
    # the line-to-line system base, smaller by sqrt(3).  Apply the conversion
    # to the live signal as well as to the frozen operating point.
    voltage_to_stamp = c(1.0 / np.sqrt(3.0))
    dvq = voltage_to_stamp*vq-vq0
    dvd = voltage_to_stamp*vd-vd0
    duq=ucapq+rac*(isq-igq); dud=ucapd+rac*(isd-igd)

    def g2l(dq, dd, q0, d0):
        return (ct*dq-st*dd+(-st*q0-ct*d0)*theta,
                st*dq+ct*dd+(ct*q0-st*d0)*theta)
    digql,digdl=g2l(igq,igd,igq0,igd0); disql,disdl=g2l(isq,isd,isq0,isd0); duql,dudl=g2l(duq,dud,uq0,ud0)
    igql0=ct*igq0-st*igd0; igdl0=st*igq0+ct*igd0
    uql0=ct*uq0-st*ud0; udl0=st*uq0+ct*ud0
    tau_m=c(p.measurement_delay)
    measurement=[(source-state)/tau_m for state,source in zip(
        (isq_m,isd_m,igq_m,igd_m,uq_m,ud_m),(disql,disdl,digql,digdl,duql,dudl))]
    dpl=c(1.5)*(uql0*igq_m+udl0*igd_m+igql0*uq_m+igdl0*ud_m)
    dql=c(1.5)*(-udl0*igq_m+uql0*igd_m+igdl0*uq_m-igql0*ud_m)
    reference_speed = one if reference_omega is None else reference_omega
    frame_speed = wb*(reference_speed-one)
    kp_i=c(p.current_kp); ki_i=c(p.current_ki)

    if p.mode == "GFOR":
        pf=vf.add_var(f"{name}.p_filt_x"); qf=vf.add_var(f"{name}.q_filt_x")
        xi_uq=vf.add_var(f"{name}.Ke_u_q"); xi_ud=vf.add_var(f"{name}.Ke_u_d")
        igd_ff=vf.add_var(f"{name}.igd_ff_x"); igq_ff=vf.add_var(f"{name}.igq_ff_x")
        omega_rad=c(p.frequency_droop_gain)*wb*pref-c(p.frequency_droop_gain/p.frequency_droop_tau)*wb*pf
        duq_ref=c(p.voltage_droop_gain)*qref+c(p.voltage_droop_gain/p.voltage_droop_tau)*qf
        kp_v,ki_v=map(c,p.voltage_pi_gains)
        isq_ref=kp_v*(duq_ref-uq_m)+ki_v*xi_uq+wb*cac*ud_m+igq_ff/c(p.current_feedforward_tau)
        isd_ref=kp_v*(-ud_m)+ki_v*xi_ud-wb*cac*uq_m+igd_ff/c(p.current_feedforward_tau)
        controls=[theta,pf,qf,xi_uq,xi_ud,igd_ff,igq_ff]
        control_eqs=[omega_rad-frame_speed,-pf/c(p.frequency_droop_tau)+dpl,
            -qf/c(p.voltage_droop_tau)-dql,duq_ref-uq_m,-ud_m,
            -igd_ff/c(p.current_feedforward_tau)+igd_m,-igq_ff/c(p.current_feedforward_tau)+igq_m]
    else:
        pll=vf.add_var(f"{name}.pll_x"); wf=vf.add_var(f"{name}.w_filt_x"); qf=vf.add_var(f"{name}.q_filt_x")
        xp=vf.add_var(f"{name}.Ke_P"); xq=vf.add_var(f"{name}.Ke_Q")
        kp_pll,ki_pll=map(c,p.pll_gains)
        omega_pu=-kp_pll*ud_m-ki_pll*pll
        dpref=pref-c(p.frequency_droop_gain/(p.frequency_droop_tau*OMEGA_BASE))*wf
        umag0=sym.sqrt(uql0*uql0+udl0*udl0+eps); dumag=(uql0*uq_m+udl0*ud_m)/umag0
        dqref=qref-c(p.voltage_droop_gain/p.voltage_droop_tau)*qf
        kp_p=c(p.current_settling_time/p.active_power_tau*230.0); ki_p=c(230.0/p.active_power_tau)
        isq_ref=kp_p*(dpref-dpl)+ki_p*xp; isd_ref=kp_p*(dqref-dql)+ki_p*xq
        controls=[pll,theta,wf,qf,xp,xq]
        control_eqs=[ud_m,wb*omega_pu-frame_speed,-wf/c(p.frequency_droop_tau)+wb*omega_pu,
                     -qf/c(p.voltage_droop_tau)+dumag,dpref-dpl,dqref-dql]

    dvcql=kp_i*(isq_ref-isq_m)+ki_i*xi_iq+wb*lc*isd_m+uq_m
    dvcdl=kp_i*(isd_ref-isd_m)+ki_i*xi_id-wb*lc*isq_m+ud_m
    vcq0=ucapq0+(rc+rac)*isq0-rac*igq0+c(p.converter_x)*isd0
    vcd0=ucapd0+(rc+rac)*isd0-rac*igd0-c(p.converter_x)*isq0
    vcql0=ct*vcq0-st*vcd0; vcdl0=st*vcq0+ct*vcd0
    dvcq=ct*dvcql+st*dvcdl+(-st*vcql0+ct*vcdl0)*theta
    dvcd=-st*dvcql+ct*dvcdl+(-ct*vcql0-st*vcdl0)*theta
    electrical=[-rtr/ltr*igq-wb*igd+(duq-dvq)/ltr,
        wb*igq-rtr/ltr*igd+(dud-dvd)/ltr,
        -(rc+rac)/lc*isq-wb*isd-ucapq/lc+dvcq/lc+rac*igq/lc,
        wb*isq-(rc+rac)/lc*isd-ucapd/lc+dvcd/lc+rac*igd/lc,
        isq/cac-igq/cac-wb*ucapd,isd/cac-igd/cac+wb*ucapq]
    if p.mode == "GFOR":
        states=[igq_m,igd_m,isq_m,isd_m,uq_m,ud_m,*controls,xi_iq,xi_id,igq,igd,isq,isd,ucapq,ucapd]
        eqs=[measurement[2],measurement[3],measurement[0],measurement[1],measurement[4],measurement[5],*control_eqs,isq_ref-isq_m,isd_ref-isd_m,*electrical]
    else:
        states=[isq_m,isd_m,igq_m,igd_m,uq_m,ud_m,*controls,xi_iq,xi_id,igq,igd,isq,isd,ucapq,ucapd]
        eqs=[*measurement,*control_eqs,isq_ref-isq_m,isd_ref-isd_m,*electrical]

    all_states=[theta_grid,*states]
    diff_vars=[dtheta_grid]+[vf.add_diff_var(f"d_{state.name}",base_var=state) for state in states]
    # STAMP current states use peak phase-to-neutral q-d pu, while VeraGrid's
    # EMT port current is instantaneous abc on the conventional RMS current
    # base.  Since I_stamp=sqrt(2/3)*I_rms and i_abc,peak=sqrt(2)*I_rms,
    # the inverse Park interface factor is sqrt(3), not 3.
    current_peak_factor = c(np.sqrt(3.0))
    iq_total=current_peak_factor*(igq0+igq)
    id_total=current_peak_factor*(igd0+igd)
    ia=vf.add_var(f"i_A_{name}",reference=VarPowerFlowReferenceType.i_A)
    ib=vf.add_var(f"i_B_{name}",reference=VarPowerFlowReferenceType.i_B)
    ic=vf.add_var(f"i_C_{name}",reference=VarPowerFlowReferenceType.i_C)
    algebraic=[ia-(iq_total*sym.sin(theta_grid)-id_total*sym.cos(theta_grid)),
               ib-(iq_total*sym.sin(theta_grid-shift)-id_total*sym.cos(theta_grid-shift)),
               ic-(iq_total*sym.sin(theta_grid+shift)-id_total*sym.cos(theta_grid+shift))]
    block=Block(state_vars=all_states,state_eqs=[wb,*eqs],diff_vars=diff_vars,
        algebraic_vars=[ia,ib,ic],algebraic_eqs=algebraic,
        init_eqs={theta_grid:z,**{state:z for state in states}},
        event_dict={pref:z,qref:z,
                    dva:c(None),dvb:c(None),dvc:c(None),**{var:c(None) for var in phase_power}},
        in_vars=[va,vb,vc],out_vars=[ia,ib,ic],name=name)
    block.diff_init_eqs = {dtheta_grid: wb}
    block.diff_init_eqs.update({diff_var: z for diff_var in diff_vars[1:]})
    block.external_mapping={VarPowerFlowReferenceType.v_A:va,VarPowerFlowReferenceType.v_B:vb,
        VarPowerFlowReferenceType.v_C:vc,VarPowerFlowReferenceType.i_A:ia,
        VarPowerFlowReferenceType.i_B:ib,VarPowerFlowReferenceType.i_C:ic,
        VarPowerFlowReferenceType.d_v_A:dva,VarPowerFlowReferenceType.d_v_B:dvb,
        VarPowerFlowReferenceType.d_v_C:dvc,
        VarPowerFlowReferenceType.P_A:phase_power[0],VarPowerFlowReferenceType.Q_A:phase_power[1],
        VarPowerFlowReferenceType.P_B:phase_power[2],VarPowerFlowReferenceType.Q_B:phase_power[3],
        VarPowerFlowReferenceType.P_C:phase_power[4],VarPowerFlowReferenceType.Q_C:phase_power[5]}
    template=EmtModelTemplate(name=name); template.tpe=DeviceType.GeneratorDevice; template.block=block
    return template
