#!/usr/bin/env python3
"""Audit actual EMT transformer equations against a static terminal equivalent."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def get_simple_linear_transformer(vf, name='linear_transformer'):
    """RL leakage and ideal tap; valid for Yg/Yg, zero shunt and zero phase shift.

    This comparison model intentionally omits the coupled magnetic core. Its
    terminal relation is the independent standard PF equivalent, not a change
    to the static transformer data.
    """
    from VeraGridEngine.Devices.Dynamic.emt_template import EmtModelTemplate
    from VeraGridEngine.enumerations import ParamPowerFlowReferenceType as P, VarPowerFlowReferenceType as V
    model = EmtModelTemplate()
    model.name = name
    block = model.block
    block.name = name
    parameters = {}
    for label, ref in [('R', P.r), ('X', P.x), ('tap', P.tap_module), ('omega', P.omega_base)]:
        var = vf.add_var(label)
        parameters[label] = var
        block.parameters[var] = vf.add_const(None)
        block.api_obj_mapping[ref] = var
    inputs = [vf.add_var(f'vf_{ph}', reference=ref) for ph, ref in zip('ABC', [V.vf_A,V.vf_B,V.vf_C])]
    inputs += [vf.add_var(f'vt_{ph}', reference=ref) for ph, ref in zip('ABC', [V.vt_A,V.vt_B,V.vt_C])]
    current = [vf.add_var(f'i_leak_{ph}') for ph in 'ABC']
    ifrom = [vf.add_var(f'if_{ph}', reference=ref) for ph, ref in zip('ABC', [V.if_A,V.if_B,V.if_C])]
    ito = [vf.add_var(f'it_{ph}', reference=ref) for ph, ref in zip('ABC', [V.it_A,V.it_B,V.it_C])]
    tap, resistance, reactance, omega = (parameters[k] for k in ('tap','R','X','omega'))
    block.in_vars = inputs
    block.state_vars = current
    block.diff_vars = [vf.add_diff_var(f'di_{ph}', base_var=var) for ph, var in zip('ABC',current)]
    block.state_eqs = [omega/(tap*tap*reactance)*(inputs[i]-tap*inputs[i+3]-tap*tap*resistance*current[i]) for i in range(3)]
    block.algebraic_vars = ifrom + ito
    block.algebraic_eqs = [ifrom[i]-current[i] for i in range(3)] + [ito[i]+tap*current[i] for i in range(3)]
    block.out_vars = ifrom + ito
    block.init_eqs = {current[i]: ifrom[i] for i in range(3)}
    block.diff_init_eqs = dict(zip(block.diff_vars,block.state_eqs))
    block.external_mapping = {var.ref:var for var in inputs+ifrom+ito}
    return model


def assert_simple_compatible(transformer):
    """Reject data that this deliberately restricted comparison cannot represent."""
    from VeraGridEngine.enumerations import WindingType
    if transformer.conn_f != WindingType.GroundedStar or transformer.conn_t != WindingType.GroundedStar:
        raise ValueError('Simple comparison requires grounded-star/grounded-star windings')
    if transformer.vector_group_number or any(abs(getattr(transformer,k)) > 1e-12 for k in ('G','B','Pcu','Pfe','I0','tap_phase')):
        raise ValueError('Simple comparison requires zero shunt, core losses and phase shift')
    if transformer.X <= 0 or transformer.tap_module <= 0:
        raise ValueError('Positive leakage reactance and tap are required')


def descriptor(block, grid, transformer, overrides=None):
    """Differentiate actual symbolic residuals, retaining all multilinear auxiliaries."""
    from VeraGridEngine.Devices.Dynamic.static_parameter_mapping import assign_static_api_object_mapping_for_device
    mapping = {}
    assign_static_api_object_mapping_for_device(grid, transformer, block, mapping)
    equations = [-eq for eq in block.state_eqs] + list(block.algebraic_eqs)
    variables = list(block.state_vars) + list(block.algebraic_vars)
    inputs = list(block.in_vars)
    all_vars = {var.uid:var for eq in equations+list(block.event_dict.values())+list(block.init_eqs.values())
                for var in eq.get_vars()}
    bindings = {uid:0. for uid in all_vars}
    bindings.update({var.uid:float(value.value) for var,value in mapping.items()})
    for var,value in block.parameters.items():
        if var not in mapping and value.value is not None:
            bindings[var.uid] = float(value.value)
    for var,expr in block.event_dict.items():
        bindings[var.uid] = float(expr.eval_uid(bindings))
    if overrides:
        for var in block.event_dict:
            if var.name in overrides:
                bindings[var.uid] = float(overrides[var.name])
    # Origin is a consistent operating point for a linear core. Initialize
    # auxiliary sqrt variables on their positive branch before differentiating.
    for var,expr in block.init_eqs.items():
        bindings[var.uid] = float(expr.eval_uid(bindings))
    residual = np.array([eq.eval_uid(bindings) for eq in equations])
    jacobian = np.zeros((len(equations),len(variables)+len(inputs)))
    for i,eq in enumerate(equations):
        uids = {v.uid for v in eq.get_vars()}
        for j,var in enumerate(variables+inputs):
            if var.uid in uids:
                jacobian[i,j] = float(eq.diff(var).eval_uid(bindings))
    mass = np.zeros((len(equations),len(variables)))
    mass[:len(block.state_vars),:len(block.state_vars)] = np.eye(len(block.state_vars))
    outputs = np.zeros((6,len(variables)))
    by_uid = {v.uid:i for i,v in enumerate(variables)}
    for i,var in enumerate(block.out_vars[:6]):
        outputs[i,by_uid[var.uid]] = 1.
    return mass,jacobian[:,:len(variables)],jacobian[:,len(variables):],outputs,bindings,float(np.max(abs(residual)))


def terminal_admittance(model, frequency):
    mass,jacobian,forcing,outputs,*_ = model
    matrix=jacobian+2j*np.pi*frequency*mass
    row_scale=np.max(abs(matrix),axis=1)
    equilibrated=matrix/row_scale[:,None]
    col_scale=np.max(abs(equilibrated),axis=0)
    response=la.solve(equilibrated/col_scale[None,:],forcing/row_scale[:,None])/col_scale[:,None]
    return -outputs @ response


def terminated_poles(model, resistance=1.):
    """Ground both ports through positive resistors; retain finite descriptor poles."""
    mass,jacobian,forcing,outputs,*_ = model
    n=len(jacobian)
    pencil = np.block([[jacobian,forcing],[resistance*outputs,np.eye(6)]])
    extended_mass = np.zeros_like(pencil)
    extended_mass[:n,:n] = mass
    poles=la.eigvals(-pencil,extended_mass)
    return poles[np.isfinite(poles)]


def static_admittance(transformer):
    y=1/complex(transformer.R,transformer.X)
    tap=transformer.tap_module
    return np.kron(np.array([[y/tap**2,-y/tap],[-y/tap,y]]),np.eye(3))


def audit_transformer(grid,transformer):
    from VeraGridEngine.Devices.Dynamic.var_factory import VarFactory
    from VeraGridEngine.Templates.Emt.xfmr_emt_multilinear_template import get_xfmr_emt_template_multilinear
    assert_simple_compatible(transformer)
    start=time.perf_counter()
    coupled_block=get_xfmr_emt_template_multilinear(VarFactory()).block
    coupled=descriptor(coupled_block,grid,transformer)
    simple=descriptor(get_simple_linear_transformer(VarFactory()).block,grid,transformer)
    uncoupled=descriptor(coupled_block,grid,transformer,{'xfmr_yoke_area_rel':1e3,'xfmr_outer_leg_area_rel':1e3})
    target=static_admittance(transformer)
    yc,ys=(terminal_admittance(model,grid.fBase) for model in (coupled,simple))
    # Actual model is linear-core by default. Check power in arbitrary phase
    # directions, not only in the balanced positive-sequence subspace.
    min_passive={}
    for label,model in [('coupled',coupled),('simple',simple)]:
        min_passive[label]=float(min(np.linalg.eigvalsh((y+y.conj().T)/2).min()
            for y in [terminal_admittance(model,f) for f in [grid.fBase*.1,grid.fBase,grid.fBase*10]]))
    poles={label:terminated_poles(model) for label,model in [('coupled',coupled),('simple',simple)]}
    names={v.name:v for v in coupled_block.event_dict}
    positive=np.exp(-2j*np.pi*np.arange(3)/3)
    balanced=np.column_stack((np.r_[positive,np.zeros(3)],np.r_[np.zeros(3),positive]))
    difference=yc-ys
    core_shunt=yc[:3,:3]+yc[:3,3:]/transformer.tap_module
    independent_core=terminal_admittance(uncoupled,grid.fBase)
    return dict(name=transformer.name,tap=transformer.tap_module,R=transformer.R,X=transformer.X,
        coupled_origin_residual=coupled[-1],simple_origin_residual=simple[-1],
        linear_core_enabled=bool(coupled[-2][names['xfmr_use_linear_core'].uid]),
        core_L_pu=coupled[-2][names['xfmr_core_linear_l_pu'].uid],
        simple_vs_static_max_abs=float(np.max(abs(ys-target))),
        coupled_vs_static_max_abs=float(np.max(abs(yc-target))),
        balanced_coupled_vs_simple_max_abs=float(np.max(abs(difference@balanced))),
        core_shunt_susceptance_eigenvalues=np.linalg.eigvalsh(-core_shunt.imag).tolist(),
        return_path_geometry_multiplier=1000,
        balanced_return_path_coupling_effect=float(np.max(abs((yc-independent_core)@balanced))),
        arbitrary_phase_return_path_coupling_effect=float(np.max(abs(yc-independent_core))),
        minimum_admittance_hermitian_eigenvalue=min_passive,
        largest_finite_terminated_pole_real={label:float(p.real.max()) for label,p in poles.items()},
        terminated_poles={label:[[float(v.real),float(v.imag)] for v in p] for label,p in poles.items()},
        elapsed_seconds=time.perf_counter()-start)


def main():
    import VeraGridEngine.api as vge
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'trunk/benchmarks/transformer_comparison')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for label,path in [('ieee39',ROOT/'Grids_and_profiles/grids/IEEE39.gridcal'),('ieee118',ROOT/'src/tests/data/grids/RAW/IEEE 118 Bus.raw')]:
        grid=vge.open_file(str(path))
        record=dict(grid=label,frequency_hz=grid.fBase,transformer_count=len(grid.transformers2w),transformers=[])
        for transformer in grid.transformers2w:
            result=audit_transformer(grid,transformer);record['transformers'].append(result)
            print(label,transformer.name,json.dumps({k:result[k] for k in ['simple_vs_static_max_abs','coupled_vs_static_max_abs','largest_finite_terminated_pole_real']}),flush=True)
        records.append(record)
        (args.output/'terminal_audit.json').write_text(json.dumps(records,indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
