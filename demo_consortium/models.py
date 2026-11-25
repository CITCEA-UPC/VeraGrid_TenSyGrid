"""Reduced-order multilinear SG/VSM sources for the consortium network demos.

These are classical machines and averaged grid-forming voltage sources, not
full manufacturer controllers. RMS and EMT share the mechanical/droop controls.
"""
from __future__ import annotations

import numpy as np

from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Var
from VeraGridEngine.Utils.Symbolic.symbolic_ml import ml_hard_sat
from VeraGridEngine.enumerations import VarPowerFlowReferenceType as Ref


def source_model(vf, name: str, voltage: complex, power: complex, rating: float,
                 base: float, frequency: float, ibr: bool, emt: bool,
                 limiter: bool = False, command: float | None = None) -> Block:
    """Build a PF-seeded lifted classical SG or averaged virtual-synchronous IBR."""
    ratio = rating/base
    resistance, reactance = .002/ratio, .2/ratio
    current = np.conj(power/voltage)
    internal = voltage+complex(resistance, reactance)*current
    e0, theta0 = abs(internal), np.angle(internal)
    pe0 = float((internal*np.conj(current)).real)
    q0 = float(power.imag)
    c, s, w, pm, pf, qf = [vf.add_var(f'{name}.{key}') for key in ('cos', 'sin', 'omega', 'pm', 'pf', 'qf')]
    e = vf.add_var(f'{name}.E')
    states = [c, s, w, pm, pf, qf, e]
    command_var = vf.add_var(f'{name}.p_command')
    pref = vf.add_var(f'{name}.p_limited')
    low, high = pe0-.15*ratio, pe0+.15*ratio
    limiter_block = None
    if limiter:
        limiter_block, limited = ml_hard_sat(vf, command_var, Const(low), Const(high), name=f'{name}_power')
    else:
        limited = command_var
    p, q, pe = [vf.add_var(f'{name}.{key}') for key in ('P', 'Q', 'Pe')]
    wb = 2*np.pi*frequency
    speed = w if emt else w-Const(1.)
    inertia = (1.5 if ibr else 5.)*ratio
    damping = (12. if ibr else 2.)*ratio
    droop = (25. if ibr else 20.)*ratio
    block = Block(name=f'{name}_{"EMT" if emt else "RMS"}_multilinear', state_vars=states,
                  diff_vars=[vf.add_diff_var(f'd_{state.name}', base_var=state) for state in states],
                  state_eqs=[-Const(wb)*speed*s, Const(wb)*speed*c,
                             (pm-pe-Const(damping)*(w-Const(1.)))/Const(2*inertia),
                             (pref-pm-Const(droop)*(w-Const(1.)))/Const(.08 if ibr else .3),
                             (pe-pf)/Const(.02), (q-qf)/Const(.02),
                             (Const(e0)-Const(.015/ratio if ibr else 0.)*(qf-Const(q0))-e)/Const(.05)],
                  algebraic_vars=[pref, p, q, pe], algebraic_eqs=[pref-limited],
                  event_dict={command_var: Const(pe0 if command is None else command)},
                  init_eqs={c: Const(np.cos(theta0)), s: Const(np.sin(theta0)), w: Const(1.),
                            pm: Const(pe0), pf: Const(pe0), qf: Const(q0), e: Const(e0),
                            pref: limited, p: Const(power.real), q: Const(q0), pe: Const(pe0)})
    if limiter_block is not None:
        block.children.append(limiter_block)
    if emt:
        voltages = [vf.add_var(f'{name}.v_{phase}', reference=ref) for phase, ref in zip('ABC', (Ref.v_A, Ref.v_B, Ref.v_C))]
        currents = [vf.add_var(f'{name}.i_{phase}', reference=ref) for phase, ref in zip('ABC', (Ref.i_A, Ref.i_B, Ref.i_C))]
        shifts = np.array([0., -2*np.pi/3, 2*np.pi/3])
        emf = [Const(np.sqrt(2.))*e*(s*Const(np.cos(angle))+c*Const(np.sin(angle))) for angle in shifts]
        for index, variable in enumerate(currents):
            block.state_vars.append(variable)
            derivative = vf.add_diff_var(f'd_{variable.name}', base_var=variable)
            block.diff_vars.append(derivative)
            rhs = Const(wb/reactance)*(emf[index]-voltages[index]-Const(resistance)*variable)
            block.state_eqs.append(rhs)
            block.init_eqs[variable] = Const(np.sqrt(2.)*(current*np.exp(1j*shifts[index])).imag)
            block.diff_init_eqs[derivative] = rhs
        block.algebraic_eqs.extend([p-sum(v*i for v, i in zip(voltages, currents))/Const(3.),
            q-((voltages[1]-voltages[2])*currents[0]+(voltages[2]-voltages[0])*currents[1]
               +(voltages[0]-voltages[1])*currents[2])/Const(3*np.sqrt(3.)),
            pe-sum(v*i for v, i in zip(emf, currents))/Const(3.)])
        block.in_vars, block.out_vars = voltages, currents
        block.external_mapping = dict(zip((Ref.v_A, Ref.v_B, Ref.v_C), voltages))
        block.external_mapping.update(dict(zip((Ref.i_A, Ref.i_B, Ref.i_C), currents)))
    else:
        vr, vi = vf.add_var(f'{name}.Vr', reference=Ref.Vr), vf.add_var(f'{name}.Vi', reference=Ref.Vi)
        ir, ii = vf.add_var(f'{name}.Ir', reference=Ref.Ir), vf.add_var(f'{name}.Ii', reference=Ref.Ii)
        er, ei = e*c, e*s
        denominator = resistance**2+reactance**2
        block.algebraic_vars.extend([ir, ii])
        block.algebraic_eqs.extend([p-vr*ir-vi*ii, q-vi*ir+vr*ii, pe-er*ir-ei*ii,
            ir-(Const(resistance)*(er-vr)+Const(reactance)*(ei-vi))/Const(denominator),
            ii-(Const(resistance)*(ei-vi)-Const(reactance)*(er-vr))/Const(denominator)])
        block.init_eqs.update({ir: Const(current.real), ii: Const(current.imag)})
        block.in_vars, block.out_vars = [vr, vi], [ir, ii]
        block.external_mapping = {Ref.Vr: vr, Ref.Vi: vi, Ref.Ir: ir, Ref.Ii: ii}
    if emt:
        block.external_mapping.update({Ref.P: p, Ref.Q: q})
    for state, rhs in zip(states, block.state_eqs[:7]):
        block.diff_init_eqs[state.diff_var] = rhs
    return block


def impedance_load_rms(vf, name: str, admittance: complex) -> Block:
    """Linear rectangular constant-impedance load with a runtime load scale."""
    vr, vi, ir, ii, scale = [vf.add_var(f'{name}.{key}') for key in ('Vr', 'Vi', 'Ir', 'Ii', 'scale')]
    vr.ref, vi.ref = Ref.Vr, Ref.Vi
    g, b = admittance.real, admittance.imag
    return Block(name=f'{name}_RMS_impedance', in_vars=[vr, vi], out_vars=[ir, ii],
        algebraic_vars=[ir, ii], algebraic_eqs=[ir+scale*Const(g)*vr-Const(b)*vi,
        ii+Const(b)*vr+scale*Const(g)*vi], event_dict={scale: Const(1.)},
        external_mapping={Ref.Vr: vr, Ref.Vi: vi, Ref.Ir: ir, Ref.Ii: ii})
