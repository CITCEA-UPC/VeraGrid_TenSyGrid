"""Network-action thermal-congestion exercises with fixed dynamic controllers.

These start at a converged RMS equilibrium with a thermal violation. They are
operating-limit repair exercises; they need not have a growing dynamic mode.
"""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from VeraGridEngine.api import open_file, save_file
from VeraGridEngine.Devices.Branches.line import Line
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver, PowerFlowOptions
from VeraGridEngine.Templates.Rms.line_rms_template import get_line_rms_template
from VeraGridEngine.Templates.Rms.line_phasor_rms_template import get_line_phasor_rms_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
from hackathon_files.basic.cases import build_network, network_specs, initialize_grid, basic_evaluator
from hackathon_files.basic.validation import equilibrium_error
from hackathon_files.src.rms_tuning import RmsTuning


def prepare_switching_grid(stable, name, phasor=False, corridor_rank=0):
    """Keep connectivity; overload one circuit and provide an open relief circuit.

    Circuit ratings are explicit educational limits, not original utility data.
    The relief circuit has the same R/X/B, so closing it shares the corridor flow.
    """
    grid=stable.copy();grid.name=name
    pf=PowerFlowDriver(grid,PowerFlowOptions(tolerance=1e-10,max_iter=100));pf.run()
    if not pf.results.converged:raise ValueError('Starting PF did not converge')
    flow=abs(pf.results.Sf[:len(grid.lines)])
    indices=[i for i in np.argsort(-flow) if grid.lines[i].active and flow[i]>1e-3]
    index=indices[corridor_rank];line=grid.lines[index]
    standby=Line(name=f'Switching relief: {line.name}',bus_from=line.bus_from,bus_to=line.bus_to,
                 r=line.R,x=line.X,b=line.B,g=line.G,rate=max(line.rate,float(flow[index])*2),active=False)
    grid.add_line(standby)
    template=get_line_phasor_rms_template if phasor else get_line_rms_template
    set_rms_model(standby,template(grid.var_factory).block,grid.var_factory)
    standby.rms_model.set_parameter_in_model('u',0.)
    line.rate=float(flow[index])/1.15
    return grid


def stationary_error(problem):
    """Check the full stationary residual for either RMS formulation."""
    from VeraGridEngine.Simulations.Rms.problems.rms_problem_multilinear import RmsProblemMultilinear
    if isinstance(problem,RmsProblemMultilinear):
        problem.build_multilinear_matrices()
        return float(np.max(abs(problem.exact_residual_from_s_phi(
            np.r_[problem.get_x0(),np.zeros(problem.get_diff_var_number())]))))
    return equilibrium_error(problem)


def validate_switching(path, initialize=initialize_grid, evaluator=None):
    """Require no-op failure and a valid switching PF/SSA/transient repair."""
    evaluator=basic_evaluator() if evaluator is None else evaluator
    snapshot=initialize(open_file(str(path)))
    assert not any((snapshot.tunable_parameters,snapshot.dispatch_options))
    assert snapshot.closable_lines and 3<=snapshot.max_structural_changes<=4
    base=evaluator.evaluate_proposal(snapshot,RmsTuning())
    assert base.thermal_violations>0,'No-op must fail the operating limits'
    proposal=RmsTuning(close_lines=tuple(x for x in snapshot.closable_lines if x.name.startswith('Switching relief:')))
    assert len(proposal.close_lines)==1 and proposal.is_accepted(snapshot)
    metrics=snapshot.test_tuning(proposal)
    assert snapshot.tests_used==1
    assert metrics.stable,metrics
    assert metrics.voltage_violations==metrics.thermal_violations==0,metrics
    assert metrics.structural_changes==1 and metrics.changes_score==1-1/snapshot.max_structural_changes
    error=stationary_error(snapshot.problem)
    assert error<=1e-6,error
    # Rebuild the post-switch equilibrium independently.
    repaired=initialize(open_file(str(path)))
    repaired_problem=evaluator.apply_tuning(repaired,proposal)
    repaired_error=stationary_error(repaired_problem)
    assert repaired_error<=1e-6,repaired_error
    return dict(file=Path(path).name,challenge_type='network_actions',
                baseline_equilibrium_error=error,baseline_max_real=base.max_eigenvalue_real,
                baseline_losses_mw=base.active_losses_mw,baseline_metrics=asdict(base),
                witness_equilibrium_error=repaired_error,witness_parameters={},
                witness_close_lines=[x.idtag for x in proposal.close_lines],
                witness_metrics={**asdict(metrics),'changes_score':metrics.changes_score},
                buses=len(snapshot.problem.grid.buses),generators=len(snapshot.problem.grid.generators),
                tunable_parameter_count=0,controller_only=False,network_only=True,
                max_structural_changes=snapshot.max_structural_changes,
                max_test_tunings='unlimited' if np.isinf(snapshot.max_test_tunings) else snapshot.max_test_tunings,
                sha256=sha256(Path(path).read_bytes()).hexdigest())


def main():
    directory=Path(__file__).with_name('testgrids');report=Path(__file__).with_name('switching_validation.json')
    previous={r['file']:r for r in json.loads(report.read_text())} if report.exists() else {}
    records=[]
    for spec in network_specs():
        name=f'basic_switching_{spec.name}_01';path=directory/f'{name}.veragrid'
        if path.exists() and path.name in previous and sha256(path.read_bytes()).hexdigest()==previous[path.name]['sha256']:
            records.append(previous[path.name]);print(name,'REUSED',flush=True);continue
        stable=build_network(spec)
        for g in stable.generators:
            for key,value in dict(M=120.,D=90.,Kp=-400.,Ki=0.).items():g.rms_model.set_parameter_in_model(key,value)
        failure=None
        for rank in range(min(5,len(stable.lines))):
            try:
                grid=prepare_switching_grid(stable,name,corridor_rank=rank);save_file(grid,str(path))
                record=validate_switching(path)
                break
            except (AssertionError,RuntimeError,ValueError) as error:
                failure=error;print(name,'retry',rank,str(error)[:300],flush=True)
        else:raise RuntimeError(f'{name}: no feasible switching repair: {failure}')
        record.update(network=spec.name,source=spec.source,formulation='dae')
        records.append(record);report.write_text(json.dumps(records,indent=2)+'\n')
        print(name,'PASS',flush=True)
    report.write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
