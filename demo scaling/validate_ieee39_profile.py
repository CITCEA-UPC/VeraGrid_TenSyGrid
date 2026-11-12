"""Validate the optional IEEE39 EMT profile using the native block period map."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import runpy
import json
import csv
from time import perf_counter
import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
audit=runpy.run_path(str(HERE/'check_ieee39_initialization.py'))
from TensyGridEngine.emt.ieee39_emt_simulation import configure_ieee39_stability_profile
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_consistent_tangent import build_consistent_emt_tangent,ProjectedEmtFloquetOperator
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.emt_floquet_operator import BlockEmtFloquetOperator
from VeraGridEngine.Simulations.EMT.initialization_emt import _emt_is_absolute_angle


def assess(context):
    problem=context['problem']; options=context['options']; solver=context['solver']
    jacobian=problem.get_floquet_jacobian_evaluator(solver.vec_jacobian)
    constants=context['constants']; events=problem.get_variable_parameter_number()
    output=[]
    for steps in (600,1200,2400):
        started=perf_counter(); stride=2400//steps
        orbit=context['orbit'][::stride]; derivatives=context['derivatives'][::stride]
        times=context['times'][::stride]; h=float(times[1]-times[0])
        tangent=build_consistent_emt_tangent(problem,orbit,derivatives,times,h,jacobian,constants,events)
        full=BlockEmtFloquetOperator(problem=problem,trajectory=orbit,h=h,
            n_states=problem.get_states_number(),method=options.integration_method,
            jac_evaluator=jacobian,static_params=constants,n_event_params=events,
            t_trajectory=times,d_trajectory=derivatives,
            initial_tangent_derivative=tangent.initial_derivative_map)
        operator=ProjectedEmtFloquetOperator(full,tangent.basis)
        size=operator.shape[0]; identity=np.eye(size); monodromy=np.empty((size,size))
        print('PROFILE FULL SPECTRUM',steps,'steps, dimension',size,flush=True)
        for first in range(0,size,16):
            monodromy[:,first:first+16]=operator.matmat(identity[:,first:first+16])
        values,vectors=la.eig(monodromy); order=np.argsort(-np.abs(values))
        values=values[order];vectors=vectors[:,order]
        phase=tangent.basis.T@context['slope'][0,context['columns'][:problem.get_states_number()]]
        alignment=np.abs(vectors.conj().T@phase)/(np.linalg.norm(vectors,axis=0)*np.linalg.norm(phase))
        errors=np.linalg.norm(monodromy@vectors-vectors*values,axis=0)/np.maximum(1.,np.abs(values))
        away_from_one=np.abs(values-1)>1e-5
        record=dict(steps=steps,max_magnitude=float(np.max(np.abs(values))),
            phase_alignment=[float(x) for x in alignment[:6]],
            phase_direction_error=float(np.linalg.norm(monodromy@phase-phase)/np.linalg.norm(phase)),
            max_magnitude_away_from_one=float(np.max(np.abs(values[away_from_one]))),
            eigenpair_residual=float(np.max(errors)),elapsed_s=perf_counter()-started,
            leading_multipliers=[[float(v.real),float(v.imag)] for v in values[:12]])
        output.append(record)
        (HERE/'ieee39_profile_validation.json').write_text(json.dumps(output,indent=2)+'\n')
        with (HERE/f'ieee39_profile_{steps}_modes.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(('real','imag','magnitude'))
            writer.writerows((v.real,v.imag,abs(v)) for v in values)
        print('PROFILE_RESULT',json.dumps(record),flush=True)
    print('Checking initialized profile by native EMT integration',flush=True)
    started=perf_counter(); simulation=solver.simulate()
    if not simulation[4]:
        raise RuntimeError('Profile integration failed')
    error=simulation[1][-1]-context['orbit'][-1]
    scale=np.maximum(1.,np.max(np.abs(context['orbit']),axis=0))
    integration=dict(samples=len(simulation[0]),elapsed_s=perf_counter()-started,
        max_scaled_deviation=float(np.max(np.abs(error)/scale)))
    print('PROFILE_INTEGRATION',json.dumps(integration),flush=True)
    (HERE/'ieee39_profile_integration.json').write_text(json.dumps(integration,indent=2)+'\n')


def profile(grid,pf):
    configure_ieee39_stability_profile(grid,pf.voltage)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        audit['main'](assessment=assess,model_profile=profile)
