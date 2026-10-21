"""Named synchronous-q-d dynamic line model for the STAMP WSCC case."""

from __future__ import annotations

from typing import Any

from .parameters import OMEGA_BASE


def build_stamp_dynamic_line_rms(vf: Any, *, resistance: float, reactance: float,
                                 shunt_susceptance: float, name: str,
                                 dynamic_shunts: bool = False) -> Any:
    """Return a series-RL line with two current states and algebraic powers."""
    from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
    from VeraGridEngine.Utils.Symbolic.block import Block
    from VeraGridEngine.Utils.Symbolic import symbolic as sym
    from VeraGridEngine.enumerations import DeviceType, VarPowerFlowReferenceType

    c = vf.add_const
    vmf = vf.add_var(f"{name}.Vmf", reference=VarPowerFlowReferenceType.Vmf)
    vaf = vf.add_var(f"{name}.Vaf", reference=VarPowerFlowReferenceType.Vaf)
    vmt = vf.add_var(f"{name}.Vmt", reference=VarPowerFlowReferenceType.Vmt)
    vat = vf.add_var(f"{name}.Vat", reference=VarPowerFlowReferenceType.Vat)
    pf = vf.add_var(f"{name}.Pf", reference=VarPowerFlowReferenceType.Pf)
    pt = vf.add_var(f"{name}.Pt", reference=VarPowerFlowReferenceType.Pt)
    qf = vf.add_var(f"{name}.Qf", reference=VarPowerFlowReferenceType.Qf)
    qt = vf.add_var(f"{name}.Qt", reference=VarPowerFlowReferenceType.Qt)
    iq = vf.add_var(f"{name}.iq")
    id_ = vf.add_var(f"{name}.id")

    vfq, vfd = vmf*sym.cos(vaf), -vmf*sym.sin(vaf)
    vtq, vtd = vmt*sym.cos(vat), -vmt*sym.sin(vat)
    half_b = c(0.0 if dynamic_shunts else shunt_susceptance/2.0)
    ifq, ifd = iq+half_b*vfd, id_-half_b*vfq
    itq, itd = -iq+half_b*vtd, -id_-half_b*vtq
    v2f = vfq*vfq+vfd*vfd+c(1e-12)
    total_ifq = (pf*vfq-qf*vfd)/v2f
    total_ifd = (pf*vfd+qf*vfq)/v2f

    state_eqs = [
        c(OMEGA_BASE/reactance)*(vfq-vtq-c(resistance)*iq-c(reactance)*id_),
        c(OMEGA_BASE/reactance)*(vfd-vtd-c(resistance)*id_+c(reactance)*iq),
    ]
    algebraic_eqs = [
        pf-(vfq*ifq+vfd*ifd),
        qf-(vfq*ifd-vfd*ifq),
        pt-(vtq*itq+vtd*itd),
        qt-(vtq*itd-vtd*itq),
    ]
    block = Block(
        state_vars=[iq, id_], state_eqs=state_eqs,
        algebraic_vars=[pf, qf, pt, qt], algebraic_eqs=algebraic_eqs,
        init_eqs={iq: total_ifq-half_b*vfd, id_: total_ifd+half_b*vfq},
        in_vars=[vmf, vaf, vmt, vat], out_vars=[pf, pt, qf, qt],
        external_mapping={
            VarPowerFlowReferenceType.Vmf: vmf, VarPowerFlowReferenceType.Vaf: vaf,
            VarPowerFlowReferenceType.Vmt: vmt, VarPowerFlowReferenceType.Vat: vat,
            VarPowerFlowReferenceType.Pf: pf, VarPowerFlowReferenceType.Pt: pt,
            VarPowerFlowReferenceType.Qf: qf, VarPowerFlowReferenceType.Qt: qt,
        }, name=name,
    )
    template = RmsModelTemplate(name=name)
    template.tpe = DeviceType.LineDevice
    template.block = block
    return template
