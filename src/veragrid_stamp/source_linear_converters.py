"""Named, parameter-dependent port of STAMP's PEAK GFOR/GFOL blocks."""

from __future__ import annotations

from typing import Any

from .bases import RMS_LL_TO_PEAK_LN
from .parameters import OMEGA_BASE, StampConverterParameters


def build_stamp_source_linear_converter(vf: Any, p: StampConverterParameters, name: str,
                                        reference_omega: Any | None = None) -> Any:
    from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
    from VeraGridEngine.Utils.Symbolic.block import Block
    from VeraGridEngine.Utils.Symbolic import symbolic as sym
    from VeraGridEngine.enumerations import DeviceType, VarPowerFlowReferenceType

    c = vf.add_const
    vm=vf.add_var(f"{name}.Vm",reference=VarPowerFlowReferenceType.Vm); va=vf.add_var(f"{name}.Va",reference=VarPowerFlowReferenceType.Va)
    pac=vf.add_var(f"{name}.P",reference=VarPowerFlowReferenceType.P); qac=vf.add_var(f"{name}.Q",reference=VarPowerFlowReferenceType.Q)
    vm0=vf.add_var(f"{name}.Vm0"); va0=vf.add_var(f"{name}.Va0"); pac0=vf.add_var(f"{name}.P0"); qac0=vf.add_var(f"{name}.Q0")
    pref=vf.add_var(f"{name}.P_ref"); qref=vf.add_var(f"{name}.Q_ref")

    k=c(RMS_LL_TO_PEAK_LN); wb=c(OMEGA_BASE); eps=c(1e-12)
    rtr=c(p.transformer_r); ltr=c(p.transformer_x/OMEGA_BASE)
    rc=c(p.converter_r); lc=c(p.converter_x/OMEGA_BASE)
    cac=c(p.capacitor_b/OMEGA_BASE); rac=c(p.damping_r)

    # Operating point in STAMP global q-d coordinates (d = -imaginary).
    vq0=k*vm0*sym.cos(va0); vd0=-k*vm0*sym.sin(va0)
    v20=vq0*vq0+vd0*vd0+eps
    # With q = real and d = -imaginary, S = 3/2 V conj(I) gives
    # Q = 3/2 (v_q i_d - v_d i_q).
    igq0=(pac0*vq0-qac0*vd0)/(c(1.5)*v20)
    igd0=(pac0*vd0+qac0*vq0)/(c(1.5)*v20)
    uq0=vq0+rtr*igq0+c(p.transformer_x)*igd0
    ud0=vd0+rtr*igd0-c(p.transformer_x)*igq0
    cap_a=rac*cac*wb
    ucapq0=(uq0-cap_a*ud0)/(c(1)+cap_a*cap_a)
    ucapd0=(cap_a*uq0+ud0)/(c(1)+cap_a*cap_a)
    isq0=igq0+cac*wb*ucapd0; isd0=igd0-cac*wb*ucapq0
    # VeraGrid's symbolic atan2 API takes Cartesian (x, y), unlike NumPy's
    # atan2(y, x).  This compiles to -atan2(u_d, u_q), as required by STAMP.
    theta0=-sym.atan2(uq0,ud0); ct=sym.cos(theta0); st=sym.sin(theta0)

    # Deviation states; physical names follow STAMP's exported state list.
    isq_m=vf.add_var(f"{name}.isq_md"); isd_m=vf.add_var(f"{name}.isd_md")
    igq_m=vf.add_var(f"{name}.igq_md"); igd_m=vf.add_var(f"{name}.igd_md")
    uq_m=vf.add_var(f"{name}.uq_md"); ud_m=vf.add_var(f"{name}.ud_md")
    theta=vf.add_var(f"{name}.etheta_x")
    igq=vf.add_var(f"{name}.ig_q"); igd=vf.add_var(f"{name}.ig_d")
    isq=vf.add_var(f"{name}.is_q"); isd=vf.add_var(f"{name}.is_d")
    ucapq=vf.add_var(f"{name}.ucap_q"); ucapd=vf.add_var(f"{name}.ucap_d")
    xi_iq=vf.add_var(f"{name}.Ke_is_q"); xi_id=vf.add_var(f"{name}.Ke_is_d")

    dvq=k*(sym.cos(va0)*(vm-vm0)-vm0*sym.sin(va0)*(va-va0))
    dvd=-k*(sym.sin(va0)*(vm-vm0)+vm0*sym.cos(va0)*(va-va0))
    duq=ucapq+rac*(isq-igq); dud=ucapd+rac*(isd-igd)

    def g2l(dq, dd, q0, d0):
        return (ct*dq-st*dd+(-st*q0-ct*d0)*theta,
                st*dq+ct*dd+(ct*q0-st*d0)*theta)

    digql,digdl=g2l(igq,igd,igq0,igd0); disql,disdl=g2l(isq,isd,isq0,isd0); duql,dudl=g2l(duq,dud,uq0,ud0)
    # STAMP's P/Q and voltage-control blocks consume the measured *local*
    # signals, so their linearization coefficients must be local as well.
    # Using global q-d operating values here introduces a spurious frame mix.
    igql0=ct*igq0-st*igd0; igdl0=st*igq0+ct*igd0
    uql0=ct*uq0-st*ud0; udl0=st*uq0+ct*ud0
    tau_m=c(p.measurement_delay)
    measurement_eqs=[(source-state)/tau_m for state,source in zip(
        (isq_m,isd_m,igq_m,igd_m,uq_m,ud_m),(disql,disdl,digql,digdl,duql,dudl))]
    dpl=c(1.5)*(uql0*igq_m+udl0*igd_m+igql0*uq_m+igdl0*ud_m)
    dql=c(1.5)*(-udl0*igq_m+uql0*igd_m+igdl0*uq_m-igql0*ud_m)

    kp_i=c(p.current_kp); ki_i=c(p.current_ki)
    reference_speed = c(1) if reference_omega is None else reference_omega
    reference_frame_speed = wb*(reference_speed-c(1))
    controller_states=[]; controller_eqs=[]
    if p.mode == "GFOR":
        pf=vf.add_var(f"{name}.p_filt_x"); qf=vf.add_var(f"{name}.q_filt_x")
        xi_uq=vf.add_var(f"{name}.Ke_u_q"); xi_ud=vf.add_var(f"{name}.Ke_u_d")
        igd_ff=vf.add_var(f"{name}.igd_ff_x"); igq_ff=vf.add_var(f"{name}.igq_ff_x")
        omega_rad=c(p.frequency_droop_gain)*wb*pref-c(p.frequency_droop_gain/p.frequency_droop_tau)*wb*pf
        duq_ref=c(p.voltage_droop_gain)*qref+c(p.voltage_droop_gain/p.voltage_droop_tau)*qf
        dud_ref=c(0)
        kp_v,ki_v=map(c,p.voltage_pi_gains)
        # MATLAB tf2ss(1, [tau_ig 1]) uses dx=-x/tau+u and y=x/tau.
        # Keep STAMP's canonical state coordinates so Jacobian entries and
        # participation factors are directly comparable.
        isq_ref=kp_v*(duq_ref-uq_m)+ki_v*xi_uq+wb*cac*ud_m+igq_ff/c(p.current_feedforward_tau)
        isd_ref=kp_v*(dud_ref-ud_m)+ki_v*xi_ud-wb*cac*uq_m+igd_ff/c(p.current_feedforward_tau)
        controller_states=[theta,pf,qf,xi_uq,xi_ud,igd_ff,igq_ff]
        controller_eqs=[omega_rad-reference_frame_speed,-pf/c(p.frequency_droop_tau)+dpl,
                        -qf/c(p.voltage_droop_tau)-dql,duq_ref-uq_m,dud_ref-ud_m,
                        -igd_ff/c(p.current_feedforward_tau)+igd_m,
                        -igq_ff/c(p.current_feedforward_tau)+igq_m]
    else:
        pll=vf.add_var(f"{name}.pll_x"); wf=vf.add_var(f"{name}.w_filt_x"); qf=vf.add_var(f"{name}.q_filt_x")
        xp=vf.add_var(f"{name}.Ke_P"); xq=vf.add_var(f"{name}.Ke_Q")
        kp_pll,ki_pll=map(c,p.pll_gains)
        omega_pu=-kp_pll*ud_m-ki_pll*pll
        dpref=pref-c(p.frequency_droop_gain/(p.frequency_droop_tau*OMEGA_BASE))*wf
        umag0=sym.sqrt(uql0*uql0+udl0*udl0+eps)
        dumag=(uql0*uq_m+udl0*ud_m)/umag0
        dqref=qref-c(p.voltage_droop_gain/p.voltage_droop_tau)*qf
        # STAMP defines Ib = Sb/Vb, hence Sb/Ib/1000 is exactly the
        # converter voltage base in kV (230 kV for this case).
        # Keep arithmetic numeric before registering symbolic constants.  An
        # add_const(Const/float) creates a Const whose *value* is an Expr; its
        # RHS happens to compile, but symbolic differentiation correctly treats
        # the outer Const as independent and loses the controller-state gain.
        voltage_base_kv=230.0
        kp_p=c(p.current_settling_time/p.active_power_tau*voltage_base_kv)
        ki_p=c(voltage_base_kv/p.active_power_tau)
        isq_ref=kp_p*(dpref-dpl)+ki_p*xp
        isd_ref=kp_p*(dqref-dql)+ki_p*xq
        controller_states=[pll,theta,wf,qf,xp,xq]
        controller_eqs=[ud_m,wb*omega_pu-reference_frame_speed,-wf/c(p.frequency_droop_tau)+wb*omega_pu,
                        -qf/c(p.voltage_droop_tau)+dumag,dpref-dpl,dqref-dql]

    dvcql=kp_i*(isq_ref-isq_m)+ki_i*xi_iq+wb*lc*isd_m+uq_m
    dvcdl=kp_i*(isd_ref-isd_m)+ki_i*xi_id-wb*lc*isq_m+ud_m
    vcq0=ucapq0+(rc+rac)*isq0-rac*igq0+c(p.converter_x)*isd0
    vcd0=ucapd0+(rc+rac)*isd0-rac*igd0-c(p.converter_x)*isq0
    vcql0=ct*vcq0-st*vcd0; vcdl0=st*vcq0+ct*vcd0
    dvcq=ct*dvcql+st*dvcdl+(-st*vcql0+ct*vcdl0)*theta
    dvcd=-st*dvcql+ct*dvcdl+(-ct*vcql0-st*vcdl0)*theta
    electrical_eqs=[
        -rtr/ltr*igq-wb*igd+(duq-dvq)/ltr,
        wb*igq-rtr/ltr*igd+(dud-dvd)/ltr,
        -(rc+rac)/lc*isq-wb*isd-ucapq/lc+dvcq/lc+rac*igq/lc,
        wb*isq-(rc+rac)/lc*isd-ucapd/lc+dvcd/lc+rac*igd/lc,
        isq/cac-igq/cac-wb*ucapd,
        isd/cac-igd/cac+wb*ucapq,
    ]
    if p.mode == "GFOR":
        states=[igq_m,igd_m,isq_m,isd_m,uq_m,ud_m,*controller_states,xi_iq,xi_id,igq,igd,isq,isd,ucapq,ucapd]
        eqs=[measurement_eqs[2],measurement_eqs[3],measurement_eqs[0],measurement_eqs[1],
             measurement_eqs[4],measurement_eqs[5],*controller_eqs,isq_ref-isq_m,isd_ref-isd_m,*electrical_eqs]
    else:
        states=[isq_m,isd_m,igq_m,igd_m,uq_m,ud_m,*controller_states,xi_iq,xi_id,igq,igd,isq,isd,ucapq,ucapd]
        eqs=[*measurement_eqs,*controller_eqs,isq_ref-isq_m,isd_ref-isd_m,*electrical_eqs]
    # The two current-controller integrators precede the six electrical states.
    assert len(states)==len(eqs)==(21 if p.mode=="GFOR" else 20)
    dpout=c(1.5)*(vq0*igq+vd0*igd+igq0*dvq+igd0*dvd)
    dqout=c(1.5)*(vq0*igd-vd0*igq+igd0*dvq-igq0*dvd)
    block=Block(state_vars=states,state_eqs=eqs,algebraic_vars=[pac,qac],algebraic_eqs=[pac-pac0-dpout,qac-qac0-dqout],
                init_eqs={state:c(0) for state in states},event_dict={vm0:vm,va0:va,pac0:pac,qac0:qac,pref:c(0),qref:c(0)},
                in_vars=[vm,va],out_vars=[pac,qac],external_mapping={VarPowerFlowReferenceType.Vm:vm,VarPowerFlowReferenceType.Va:va,
                VarPowerFlowReferenceType.P:pac,VarPowerFlowReferenceType.Q:qac},name=name)
    template=RmsModelTemplate(name=name); template.tpe=DeviceType.StaticGeneratorDevice; template.block=block
    return template
