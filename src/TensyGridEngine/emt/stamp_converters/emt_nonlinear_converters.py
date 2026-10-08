"""Three-phase EMT terminal wrapper for the nonlinear STAMP converter blocks."""

from __future__ import annotations

import numpy as np

from .nonlinear_converters import build_stamp_converter_rms
from .parameters import OMEGA_BASE, StampConverterParameters


def build_stamp_nonlinear_converter_emt(vf, params: StampConverterParameters,
                                        name: str):
    """Connect the full nonlinear q-d equations to instantaneous abc ports."""
    from VeraGridEngine.Devices.Dynamic.emt_template import EmtModelTemplate
    from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_pf_positive_sequence_init_refs
    from VeraGridEngine.Utils.Symbolic import symbolic as sym
    from VeraGridEngine.enumerations import DeviceType, VarPowerFlowReferenceType

    c = vf.add_const
    rms = build_stamp_converter_rms(vf, params, name.replace("_EMT", ""))
    block = rms.block
    vm, angle = block.in_vars
    power, reactive = block.algebraic_vars
    abc_voltage = [vf.add_var(f"v_{phase}_{name}", reference=getattr(VarPowerFlowReferenceType, f"v_{phase}"))
                   for phase in "ABC"]
    abc_derivative = [vf.add_var(f"d_v_{phase}_{name}", reference=getattr(VarPowerFlowReferenceType, f"d_v_{phase}"))
                      for phase in "ABC"]
    phase_power = [vf.add_var(f"{quantity}_{phase}_{name}", reference=getattr(VarPowerFlowReferenceType, f"{quantity}_{phase}"))
                   for phase in "ABC" for quantity in "PQ"]
    phi_v0, _, vpk0, _ = get_pf_positive_sequence_init_refs(
        v_a=abc_voltage[0], v_b=abc_voltage[1], v_c=abc_voltage[2],
        d_v_a=abc_derivative[0], d_v_b=abc_derivative[1], d_v_c=abc_derivative[2],
        p_a=phase_power[0], q_a=phase_power[1], p_b=phase_power[2],
        q_b=phase_power[3], p_c=phase_power[4], q_c=phase_power[5],
        omega_base=c(OMEGA_BASE))
    theta_grid = vf.add_var(f"{name}.theta_grid")
    dtheta_grid = vf.add_diff_var(f"d_{name}.theta_grid", base_var=theta_grid)
    shift = 2.0*np.pi/3.0
    angles = (theta_grid, theta_grid-c(shift), theta_grid+c(shift))
    vq = c(2.0/3.0)*sum(sym.sin(a)*v for a, v in zip(angles, abc_voltage))
    vd = -c(2.0/3.0)*sum(sym.cos(a)*v for a, v in zip(angles, abc_voltage))
    # EMT's Park voltage is sqrt(2) times RMS line-to-line pu; the nonlinear
    # block expects RMS line-to-line Vm and the same rotating q/d angle.
    block.algebraic_vars.extend([vm, angle])
    block.algebraic_eqs.extend([
        c(np.sqrt(2.0))*vm*sym.cos(angle)-vq,
        -c(np.sqrt(2.0))*vm*sym.sin(angle)-vd,
    ])
    block.init_eqs[vm] = vpk0/c(np.sqrt(2.0))
    block.init_eqs[angle] = phi_v0
    block.init_eqs[power] = c(params.p_pu_system)
    # The two solved Q values are the same audited operating points used by
    # the established linearized EMT wrapper.
    q0 = {"GFOR": -0.0437531356548132, "GFOL": -0.139271984262034}[params.mode]
    block.init_eqs[reactive] = c(q0)

    igq = next(var for var in block.state_vars if var.name.endswith(".ig_q"))
    igd = next(var for var in block.state_vars if var.name.endswith(".ig_d"))
    factor = c(np.sqrt(3.0))
    abc_current = [vf.add_var(f"i_{phase}_{name}", reference=getattr(VarPowerFlowReferenceType, f"i_{phase}"))
                   for phase in "ABC"]
    block.algebraic_vars.extend(abc_current)
    block.algebraic_eqs.extend([
        current-factor*(igq*sym.sin(a)-igd*sym.cos(a))
        for current, a in zip(abc_current, angles)
    ])
    original_states = list(block.state_vars)
    block.state_vars.insert(0, theta_grid)
    block.state_eqs.insert(0, c(OMEGA_BASE))
    block.diff_vars = [dtheta_grid] + [vf.add_diff_var(f"d_{var.name}", base_var=var)
                                      for var in original_states]
    block.init_eqs[theta_grid] = c(0.0)
    block.diff_init_eqs = {dtheta_grid: c(OMEGA_BASE)}
    block.in_vars = abc_voltage
    block.out_vars = abc_current
    block.external_mapping = {
        **{getattr(VarPowerFlowReferenceType, f"v_{phase}"): var
           for phase, var in zip("ABC", abc_voltage)},
        **{getattr(VarPowerFlowReferenceType, f"i_{phase}"): var
           for phase, var in zip("ABC", abc_current)},
        **{getattr(VarPowerFlowReferenceType, f"d_v_{phase}"): var
           for phase, var in zip("ABC", abc_derivative)},
        **{getattr(VarPowerFlowReferenceType, f"{quantity}_{phase}"): var
           for phase in "ABC" for quantity, var in zip("PQ", phase_power[2*(ord(phase)-65):2*(ord(phase)-65)+2])},
    }
    block.event_dict.update({var: c(None) for var in [*abc_derivative, *phase_power]})
    model = EmtModelTemplate(name=name)
    model.tpe = DeviceType.GeneratorDevice
    model.block = block
    return model
