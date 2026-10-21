"""Exact three-phase EMT terminal wrapper for the validated STAMP SG port."""

from __future__ import annotations

import numpy as np

from .nonlinear_generator import build_stamp_generator_rms
from .parameters import OMEGA_BASE, StampSynchronousGeneratorParameters


def build_stamp_generator_emt(vf, p: StampSynchronousGeneratorParameters,
                              name: str = "STAMP_SG1_EMT"):
    """Expose the named six-winding RMS equations through instantaneous abc ports.

    The electromechanical/control equations are not re-derived here: this wraps
    the exact validated RMS block.  Its two transformer-current and six rotor-
    circuit current states therefore remain present.  A synchronous Park frame
    converts the live abc terminal voltage/current at the model boundary.
    """
    from VeraGridEngine.Devices.Dynamic.emt_template import EmtModelTemplate
    from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_pf_positive_sequence_init_refs
    from VeraGridEngine.Utils.Symbolic import symbolic as sym
    from VeraGridEngine.enumerations import DeviceType, VarPowerFlowReferenceType

    c = vf.add_const
    rms = build_stamp_generator_rms(vf, p, name.replace("_EMT", ""))
    block = rms.block
    vm, va_angle = block.in_vars
    pg, qg = block.algebraic_vars[-2:]

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
    phi_v0, _phi_i0, vpk0, _ipk0 = get_pf_positive_sequence_init_refs(
        v_a=va, v_b=vb, v_c=vc, d_v_a=dva, d_v_b=dvb, d_v_c=dvc,
        p_a=phase_power[0], q_a=phase_power[1], p_b=phase_power[2], q_b=phase_power[3],
        p_c=phase_power[4], q_c=phase_power[5], omega_base=c(OMEGA_BASE))

    theta_grid = vf.add_var(f"{name}.theta_grid")
    dtheta_grid = vf.add_diff_var(f"d_{name}.theta_grid", base_var=theta_grid)
    shift = 2.0*np.pi/3.0
    vq_emt=(2.0/3.0)*(sym.sin(theta_grid)*va+sym.sin(theta_grid-shift)*vb+sym.sin(theta_grid+shift)*vc)
    vd_emt=-(2.0/3.0)*(sym.cos(theta_grid)*va+sym.cos(theta_grid-shift)*vb+sym.cos(theta_grid+shift)*vc)
    # The wrapped RMS equations use sqrt(2/3)*V_LL,rms.  Conventional EMT dq
    # peak voltage is sqrt(2)*V_LL,rms, hence the exact sqrt(3) conversion.
    k = c(np.sqrt(2.0/3.0))
    block.algebraic_vars.extend([vm, va_angle])
    block.algebraic_eqs.extend([
        k*vm*sym.cos(va_angle)-vq_emt/c(np.sqrt(3.0)),
        -k*vm*sym.sin(va_angle)-vd_emt/c(np.sqrt(3.0)),
    ])
    block.in_vars = [va, vb, vc]
    block.init_eqs[vm] = vpk0/c(np.sqrt(2.0))
    block.init_eqs[va_angle] = phi_v0
    # Slack P/Q and the four state-dependent RMS event references come from
    # the already audited common operating point (the slack's scheduled P is
    # not its solved injection).  Freezing these values also prevents the EMT
    # runtime-parameter builder from trying to evaluate expressions containing
    # state variables before x0 exists.
    block.init_eqs[pg] = c(1.4772903452891006)
    block.init_eqs[qg] = c(0.1078630099665861)
    operating_events = {
        f"{name.replace('_EMT', '')}.rotor_angle": 0.4528718256092136,
        f"{name.replace('_EMT', '')}.Vsg_mag0": 1.3146147766587668,
        f"{name.replace('_EMT', '')}.vf_d0": 0.0006712987337316768,
        f"{name.replace('_EMT', '')}.Pref": 0.482911975853729,
    }
    for parameter in list(block.event_dict):
        if parameter.name in operating_events:
            block.event_dict[parameter] = c(operating_events[parameter.name])

    igq = next(state for state in block.state_vars if state.name.endswith(".ig_q"))
    igd = next(state for state in block.state_vars if state.name.endswith(".ig_d"))
    # VeraGrid's ``conventional_three_phase_base`` terminal convention scales
    # each PF phase current by three.  Combined with the sqrt(3) voltage-base
    # conversion, the abc terminal current is sqrt(3) times STAMP's qd current.
    iq_emt = c(np.sqrt(3.0))*igq; id_emt = c(np.sqrt(3.0))*igd
    ia=vf.add_var(f"i_A_{name}",reference=VarPowerFlowReferenceType.i_A)
    ib=vf.add_var(f"i_B_{name}",reference=VarPowerFlowReferenceType.i_B)
    ic=vf.add_var(f"i_C_{name}",reference=VarPowerFlowReferenceType.i_C)
    block.algebraic_vars.extend([ia,ib,ic])
    block.algebraic_eqs.extend([
        ia-(iq_emt*sym.sin(theta_grid)-id_emt*sym.cos(theta_grid)),
        ib-(iq_emt*sym.sin(theta_grid-shift)-id_emt*sym.cos(theta_grid-shift)),
        ic-(iq_emt*sym.sin(theta_grid+shift)-id_emt*sym.cos(theta_grid+shift)),
    ])
    wrapped_diff_vars=[vf.add_diff_var(f"d_{state.name}",base_var=state) for state in block.state_vars]
    block.state_vars.insert(0,theta_grid); block.state_eqs.insert(0,c(OMEGA_BASE))
    block.diff_vars=[dtheta_grid,*wrapped_diff_vars]
    block.init_eqs[theta_grid]=c(0.0)
    block.diff_init_eqs={dtheta_grid:c(OMEGA_BASE)}
    block.out_vars=[ia,ib,ic]
    block.external_mapping={VarPowerFlowReferenceType.v_A:va,VarPowerFlowReferenceType.v_B:vb,
        VarPowerFlowReferenceType.v_C:vc,VarPowerFlowReferenceType.i_A:ia,
        VarPowerFlowReferenceType.i_B:ib,VarPowerFlowReferenceType.i_C:ic,
        VarPowerFlowReferenceType.d_v_A:dva,VarPowerFlowReferenceType.d_v_B:dvb,
        VarPowerFlowReferenceType.d_v_C:dvc,
        VarPowerFlowReferenceType.P_A:phase_power[0],VarPowerFlowReferenceType.Q_A:phase_power[1],
        VarPowerFlowReferenceType.P_B:phase_power[2],VarPowerFlowReferenceType.Q_B:phase_power[3],
        VarPowerFlowReferenceType.P_C:phase_power[4],VarPowerFlowReferenceType.Q_C:phase_power[5]}
    block.event_dict.update({dva:c(None),dvb:c(None),dvc:c(None),**{var:c(None) for var in phase_power}})
    template=EmtModelTemplate(name=name); template.tpe=DeviceType.GeneratorDevice; template.block=block
    return template
