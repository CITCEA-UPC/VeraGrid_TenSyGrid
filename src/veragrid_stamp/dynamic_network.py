"""Dynamic bus-capacitor and RL-load models used by STAMP's RMS network."""

from __future__ import annotations

from typing import Any

from .parameters import OMEGA_BASE


def build_dynamic_rl_load_rms(vf: Any, *, p_pu: float, q_pu: float, name: str) -> Any:
    from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
    from VeraGridEngine.Utils.Symbolic.block import Block
    from VeraGridEngine.Utils.Symbolic import symbolic as sym
    from VeraGridEngine.enumerations import DeviceType, VarPowerFlowReferenceType
    c = vf.add_const
    vm = vf.add_var(f"{name}.Vm", reference=VarPowerFlowReferenceType.Vm)
    va = vf.add_var(f"{name}.Va", reference=VarPowerFlowReferenceType.Va)
    pl = vf.add_var(f"{name}.Pl", reference=VarPowerFlowReferenceType.P)
    ql = vf.add_var(f"{name}.Ql", reference=VarPowerFlowReferenceType.Q)
    iq, id_ = vf.add_var(f"{name}.ilq"), vf.add_var(f"{name}.ild")
    vq, vd = vm*sym.cos(va), -vm*sym.sin(va)
    denominator = p_pu*p_pu+q_pu*q_pu
    conductance, inverse_inductance = (vf.add_var(f"{name}.G0"),
                                       vf.add_var(f"{name}.invL0"))
    v2 = vq*vq+vd*vd+c(1e-12)
    # VeraGrid injection-device P/Q are positive into the network; an RL load
    # therefore reports the negative of its consumed power.
    consumed_iq = (-pl*vq+ql*vd)/v2; consumed_id = (-pl*vd-ql*vq)/v2
    iq0 = consumed_iq-conductance*vq; id0 = consumed_id-conductance*vd
    total_iq, total_id = conductance*vq+iq, conductance*vd+id_
    block = Block(
        state_vars=[iq, id_],
        state_eqs=[-c(OMEGA_BASE)*id_+inverse_inductance*vq,
                   c(OMEGA_BASE)*iq+inverse_inductance*vd],
        algebraic_vars=[pl, ql],
        algebraic_eqs=[pl+(vq*total_iq+vd*total_id),
                       ql+(vq*total_id-vd*total_iq)],
        init_eqs={iq: iq0, id_: id0},
        event_dict={conductance:c(p_pu)*vm**c(-2.0),
                    inverse_inductance:c(OMEGA_BASE*q_pu)*vm**c(-2.0)},
        in_vars=[vm, va], out_vars=[pl, ql],
        external_mapping={VarPowerFlowReferenceType.Vm: vm, VarPowerFlowReferenceType.Va: va,
                          VarPowerFlowReferenceType.P: pl, VarPowerFlowReferenceType.Q: ql},
        name=name)
    template=RmsModelTemplate(name=name); template.tpe=DeviceType.LoadDevice; template.block=block
    return template


def build_dynamic_bus_capacitor_rms(vf: Any, *, susceptance: float, name: str) -> Any:
    from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
    from VeraGridEngine.Utils.Symbolic.block import Block
    from VeraGridEngine.Utils.Symbolic import symbolic as sym
    from VeraGridEngine.enumerations import DeviceType, VarPowerFlowReferenceType
    c = vf.add_const
    vm = vf.add_var(f"{name}.Vm", reference=VarPowerFlowReferenceType.Vm)
    va = vf.add_var(f"{name}.Va", reference=VarPowerFlowReferenceType.Va)
    pl = vf.add_var(f"{name}.P", reference=VarPowerFlowReferenceType.P)
    ql = vf.add_var(f"{name}.Q", reference=VarPowerFlowReferenceType.Q)
    vq, vd = vf.add_var(f"{name}.vc_q"), vf.add_var(f"{name}.vc_d")
    bus_q, bus_d = vm*sym.cos(va), -vm*sym.sin(va)
    v2 = vq*vq+vd*vd+c(1e-12)
    iq = (-pl*vq+ql*vd)/v2; id_ = (-pl*vd-ql*vq)/v2
    block = Block(
        state_vars=[vq, vd],
        state_eqs=[c(OMEGA_BASE/susceptance)*iq-c(OMEGA_BASE)*vd,
                   c(OMEGA_BASE/susceptance)*id_+c(OMEGA_BASE)*vq],
        algebraic_vars=[pl, ql], algebraic_eqs=[vq-bus_q, vd-bus_d],
        init_eqs={vq: bus_q, vd: bus_d}, in_vars=[vm, va], out_vars=[pl, ql],
        external_mapping={VarPowerFlowReferenceType.Vm: vm, VarPowerFlowReferenceType.Va: va,
                          VarPowerFlowReferenceType.P: pl, VarPowerFlowReferenceType.Q: ql},
        name=name)
    template=RmsModelTemplate(name=name); template.tpe=DeviceType.LoadDevice; template.block=block
    return template
