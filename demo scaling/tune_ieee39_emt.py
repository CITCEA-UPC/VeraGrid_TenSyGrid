"""Explore EMT damping/load parameters at the unchanged IEEE39 PF orbit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import runpy
import json
import csv
import argparse
from time import perf_counter
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
audit=runpy.run_path(str(HERE/'check_ieee39_initialization.py'))
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_consistent_tangent import build_consistent_emt_tangent,ProjectedEmtFloquetOperator
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_floquet_operator import BlockEmtFloquetOperator


def assess(context):
    problem=context['problem']; driver=context['driver']; options=context['options']
    full_orbit=context['orbit']; full_derivatives=context['derivatives']
    orbit=full_orbit[::4]; derivatives=full_derivatives[::4]; times=context['times'][::4]
    h=float(times[1]-times[0]); nstates=problem.get_states_number()
    runtime=problem.def_event_params_fn(problem.event_params_values.copy(),0.)
    constants=context['constants']; baseline=np.r_[runtime,constants]
    parameters=problem.get_variable_parameters()+problem.get_constant_parameters()
    parameter_index={v.uid:i for i,v in enumerate(parameters)}
    original_jacobian=problem.get_floquet_jacobian_evaluator(context['solver'].vec_jacobian)
    def find_vars(device):
        return {v.name:v for b in device.emt_model.get_all_blocks()
                for v in list(b.event_dict)+list(b.parameters)+b.algebraic_vars+b.state_vars+b.in_vars}
    load_controls=[]
    for load in problem.grid.loads:
        variables=find_vars(load)
        # Match the baseline power exactly at the actual PF voltage, including
        # the existing regularization in r=Vm/V0.
        r=problem.get_x0()[problem.get_var_idx(variables['rA'])]
        load_controls.append((variables,r))
    machines=[find_vars(gen) for gen in problem.grid.generators]
    field_controls=[]
    for gen,variables in zip(problem.grid.generators,machines):
        bus_variables=find_vars(gen.bus)
        voltage_columns=np.array([problem.get_var_idx(bus_variables['v_'+phase]) for phase in 'ABC'])
        nominal=float(np.sqrt(np.sum(problem.get_x0()[voltage_columns]**2)/3))
        field_row=list(problem.get_state_vars()).index(variables['psi_f'])
        field_controls.append((field_row,voltage_columns,nominal,
            baseline[parameter_index[variables['omega_base'].uid]],
            baseline[parameter_index[variables['v_f'].uid]]))
    args=globals().get('ARGS')
    stride=2400//args.steps if args else 4
    orbit=full_orbit[::stride]; derivatives=full_derivatives[::stride]; times=context['times'][::stride]
    h=float(times[1]-times[0])
    output=[]
    cases=[('baseline',0.,None,None),('damping_025',0.,None,.25),
           ('z10',.1,None,None),('z25',.25,None,None),
           ('sogi2',0.,2.,None),('sogi4',0.,4.,None),
           ('z25_sogi2_damping025',.25,2.,.25),
           ('z35_damping025',.35,None,.25),('z50_damping025',.5,None,.25),
           ('z75_damping025',.75,None,.25)]
    cases=[(*case,0.) for case in cases]
    cases.extend([('z25_avr2',.25,None,.25,2.),('z25_avr10',.25,None,.25,10.),
                  ('z10_avr10',.1,None,.25,10.)])
    if args and args.cases:
        selected=set(args.cases.split(','))
        cases=[case for case in cases if case[0] in selected]
        if {case[0] for case in cases}!=selected:
            raise ValueError('Unknown case name')
    for name,z,k,damping,excitation in cases:
        started=perf_counter(); tuned=baseline.copy()
        for variables,r in load_controls:
            for key,value in {'a1':z/r**2,'a2':0.,'a3':1-z,
                              'a4':z/r**2,'a5':0.,'a6':1-z}.items():
                tuned[parameter_index[variables[key].uid]]=value
            if k is not None:
                tuned[parameter_index[variables['k_sogi'].uid]]=k
        if damping is not None:
            for variables in machines:
                inertia=baseline[parameter_index[variables['H'].uid]]
                tuned[parameter_index[variables['D'].uid]]=2*inertia*damping
        residual=np.empty(len(context['equations'])); worst=0.
        for idx in np.linspace(0,len(full_orbit)-1,25,dtype=int):
            ordered=full_orbit[idx,context['columns']]
            context['tape'].residual(ordered,tuned,ordered,np.empty(0),1.,residual,ordered)
            residual[:nstates]-=context['slope'][idx,context['columns'][:nstates]]
            if excitation:
                for row,cols,reference,omega,field_voltage in field_controls:
                    measured=float(np.sqrt(np.sum(full_orbit[idx,cols]**2)/3))
                    residual[row]-=omega*field_voltage*excitation*(1-measured/reference)
            if not np.all(np.isfinite(residual)):
                raise RuntimeError('Nonfinite tuned orbit')
            worst=max(worst,float(np.max(np.abs(residual))))
        if worst>1e-6:
            raise RuntimeError(f'{name} changes operating point: residual {worst}')
        changed=np.flatnonzero(tuned!=baseline)
        def jacobian(**kwargs):
            values=np.array(kwargs['params'],copy=True)
            values[changed]=tuned[changed]
            kwargs['params']=values
            result=original_jacobian(**kwargs)
            if excitation:
                rows=[];cols_out=[];data=[]
                for row,cols,reference,omega,field_voltage in field_controls:
                    voltages=kwargs['states'][cols]
                    measured=float(np.sqrt(np.sum(voltages**2)/3))
                    rows.extend([row]*3);cols_out.extend(cols)
                    data.extend(omega*field_voltage*excitation/reference*voltages/(3*measured))
                result=result+sp.csc_matrix((data,(rows,cols_out)),shape=result.shape)
            return result
        print('CASE',name,'same-orbit residual',worst,flush=True)
        tangent=build_consistent_emt_tangent(problem,orbit,derivatives,times,h,jacobian,constants,len(runtime))
        full=BlockEmtFloquetOperator(problem=problem,trajectory=orbit,h=h,n_states=nstates,
            method=options.integration_method,jac_evaluator=jacobian,static_params=constants,
            n_event_params=len(runtime),t_trajectory=times,d_trajectory=derivatives,
            initial_tangent_derivative=tangent.initial_derivative_map)
        operator=ProjectedEmtFloquetOperator(full,tangent.basis)
        size=operator.shape[0]
        monodromy=np.empty((size,size))
        for first in range(0,size,16):
            monodromy[:,first:first+16]=operator.matmat(np.eye(size)[:,first:first+16])
        values,vectors=la.eig(monodromy)
        order=np.argsort(-np.abs(values)); values=values[order]; vectors=vectors[:,order]
        errors=np.linalg.norm(monodromy@vectors-vectors*values,axis=0)/np.maximum(1.,np.abs(values))
        phase_direction=tangent.basis.T@context['slope'][0,context['columns'][:nstates]]
        alignment=np.abs(vectors.conj().T@phase_direction)/(np.linalg.norm(vectors,axis=0)*np.linalg.norm(phase_direction))
        phase_error=float(np.linalg.norm(monodromy@phase_direction-phase_direction)/np.linalg.norm(phase_direction))
        record=dict(phase_alignment=[float(v) for v in alignment[:12]],phase_direction_error=phase_error,name=name,z_fraction=z,sogi_k=k,damping_per_s=damping,excitation_gain=excitation,
            steps=args.steps if args else 600, max_magnitude=float(np.max(np.abs(values))),unstable_modes=int(np.count_nonzero(np.abs(values)>1.00001)),
            orbit_residual=worst,eigenpair_residual=float(np.max(errors)),elapsed_s=perf_counter()-started,
            leading_multipliers=[[float(v.real),float(v.imag)] for v in values[:12]])
        output.append(record)
        (HERE/f'ieee39_tuning_scan_{args.tag if args else "screen"}.json').write_text(json.dumps(output,indent=2)+'\n')
        with (HERE/f'ieee39_tuning_{name}_{args.steps if args else 600}_modes.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(('real','imag','magnitude'))
            writer.writerows((v.real,v.imag,abs(v)) for v in values)
        print('RESULT',json.dumps(record),flush=True)
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases')
    parser.add_argument('--steps',type=int,choices=(600,1200,2400),default=600)
    parser.add_argument('--tag',default='screen')
    ARGS=parser.parse_args()
    with threadpool_limits(limits=1):
        audit['main'](assessment=assess)
