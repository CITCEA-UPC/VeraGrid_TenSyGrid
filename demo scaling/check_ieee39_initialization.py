"""Audit IEEE39 initialization and evaluate SSA on its analytical balanced orbit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import runpy
from pathlib import Path
from time import perf_counter
import json

runner = runpy.run_path(str(Path(__file__).with_name('run_ieee39_ssa.py')))
import numpy as np
from threadpoolctl import threadpool_limits
from TensyGridEngine.emt.ieee9_emt_simulation import attach_emt_models
from TensyGridEngine.emt.ieee39_emt_simulation import configure_ieee39_simple_machines, build_emt_options, build_power_flow_options
from TensyGridEngine.emt.emt_power_flow import run_emt_power_flows
from VeraGridEngine.Simulations.EMT.initialization_emt import _emt_phase_groups, _emt_is_absolute_angle
from VeraGridEngine.Simulations.EMT.solvers.StructuralVectorizedSolver import SparseADJacobian, StructuralVectorizedSolver
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import SmallSignalStabilityEmtDriver
from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import SmallSignalStabilityEmtOptions
from VeraGridEngine.enumerations import EmtSolverTypes


def main(assessment=None, model_profile=None):
    started = perf_counter()
    grid = runner['static_grid_with_rms_dispatch']()
    attach_emt_models(grid)
    configure_ieee39_simple_machines(grid)
    pf, pf3 = run_emt_power_flows(grid, build_power_flow_options())
    if model_profile is not None:
        model_profile(grid, pf)
    options = build_emt_options()
    options.solver_type = EmtSolverTypes.StructuralAD
    steps = 2400
    period = 1/grid.fBase
    options.time_step = period/steps
    options.init_newton_max_iter = 50
    ss = SmallSignalStabilityEmtOptions(k=3, target_period=period,
        ss_assessment_time=period, consistent_dae_tangent=True)
    print('Building initialized IEEE39 problem', flush=True)
    driver = SmallSignalStabilityEmtDriver(grid=grid, emt_options=options,
        sss_options=ss, pf_results=pf, pf_results_3ph=pf3)
    problem = driver.problem
    states = problem.get_state_vars()
    variables = states + problem.get_algebraic_vars()
    positions = {v.uid: problem.get_var_idx(v) for v in variables}
    equations = problem.get_state_eqs() + problem.get_algebraic_eqs()
    x0 = problem.get_x0()
    omega = 2*np.pi*grid.fBase
    times = np.linspace(0, period, steps+1)
    phase = omega*times
    orbit = np.tile(x0, (steps+1, 1))
    slope = np.zeros_like(orbit)
    devices = [device for group in (grid.buses,grid.lines,grid.transformers2w,
        grid.generators,grid.loads,grid.shunts) for device in group]
    groups = {}
    for device in devices:
        if device.emt_model is None:
            continue
        for block in device.emt_model.get_all_blocks():
            for group in _emt_phase_groups(block.state_vars + block.algebraic_vars):
                if group[0].name[0].islower() and all(v.uid in positions for v in group):
                    groups[tuple(v.uid for v in group)] = group
    for a,b,c in groups.values():
        for var, plus, minus in ((a,c,b),(b,a,c),(c,b,a)):
            col = positions[var.uid]
            quadrature = (x0[positions[plus.uid]]-x0[positions[minus.uid]])/np.sqrt(3)
            orbit[:,col] = x0[col]*np.cos(phase)+quadrature*np.sin(phase)
            slope[:,col] = omega*(-x0[col]*np.sin(phase)+quadrature*np.cos(phase))
    for var in states:
        if _emt_is_absolute_angle(var):
            orbit[:,positions[var.uid]] += phase
            slope[:,positions[var.uid]] = omega
    columns = np.array([positions[v.uid] for v in variables])
    tape = SparseADJacobian(equations, variables,
        problem.get_variable_parameters()+problem.get_constant_parameters(), [],
        options.integration_method, tape_only=True)._tape
    constants = np.array([p.value for p in problem.get_parameters_values()])
    residual = np.empty(len(equations))
    worst_state = worst_alg = runtime_change = 0.0
    runtime0 = problem.def_event_params_fn(problem.event_params_values.copy(), 0.)
    direct_worst = 0.0
    check_indices = set(np.linspace(0,steps,19,dtype=int))
    for k,t in enumerate(times):
        runtime = problem.def_event_params_fn(problem.event_params_values.copy(), float(t))
        runtime_change = max(runtime_change, float(np.max(np.abs(runtime-runtime0))))
        parameters = np.r_[runtime,constants]
        ordered = orbit[k,columns]
        tape.residual(ordered, parameters, ordered, np.empty(0), 1., residual, ordered)
        residual[:len(states)] -= slope[k,columns[:len(states)]]
        if not np.all(np.isfinite(residual)):
            raise RuntimeError(f'Nonfinite original-equation residual at t={t}')
        worst_state = max(worst_state, float(np.max(np.abs(residual[:len(states)]))))
        worst_alg = max(worst_alg, float(np.max(np.abs(residual[len(states):]))))
        if k in check_indices:
            bindings = {v.uid:float(orbit[k,positions[v.uid]]) for v in variables}
            bindings.update({v.uid:float(value) for v,value in zip(
                problem.get_variable_parameters()+problem.get_constant_parameters(),parameters)})
            bindings[problem.glob_time.uid] = float(t)
            independent = np.array([eq.eval_uid(bindings) for eq in equations])
            independent[:len(states)] -= slope[k,columns[:len(states)]]
            direct_worst = max(direct_worst,float(np.max(np.abs(independent))))
    derivatives = np.column_stack([slope[:,positions[d.base_var.uid]] for d in problem.get_diff_vars()])
    dx_error = problem.get_dx0()-derivatives[0]
    metrics = dict(samples=len(times), continuous_state_residual=worst_state,
        algebraic_residual=worst_alg, independent_expression_residual=direct_worst,
        initial_derivative_error=float(np.max(np.abs(dx_error))),
        runtime_parameter_change=runtime_change,
        runtime_seed_error=float(np.max(np.abs(runtime0-problem.event_params_values))))
    saved_capture = Path(__file__).with_name('ieee39_emt_capture_2400.npz')
    if saved_capture.exists() and model_profile is None:
        with np.load(saved_capture) as capture:
            numerical = capture['trajectory']
        reference = orbit.copy()
        for var in states:
            if _emt_is_absolute_angle(var):
                reference[:,positions[var.uid]] += 4*np.pi
        scale = np.maximum(1.,np.max(np.abs(reference),axis=0))
        metrics['integrated_last_period_scaled_deviation'] = float(np.max(np.abs(numerical-reference)/scale))
    output = Path(__file__).with_name('ieee39_initialization_audit.json' if assessment is None else 'ieee39_tuning_baseline_audit.json')
    output.write_text(json.dumps(metrics,indent=2)+'\n')
    print('INITIALIZATION_AUDIT', json.dumps(metrics), flush=True)
    if max(worst_state,worst_alg,direct_worst,float(np.max(np.abs(dx_error))),
           runtime_change,metrics['runtime_seed_error']) > 1e-6:
        raise RuntimeError('Initialization audit failed')
    print('Evaluating native SSA on analytical orbit, without time integration', flush=True)
    solver = StructuralVectorizedSolver(problem=problem,t0=0.,t_end=period,
        h=options.time_step,method=options.integration_method)
    static = constants
    if assessment is not None:
        return assessment(locals())
    def exact_capture(self,h,verbose=0):
        return orbit,derivatives,times,problem.get_floquet_jacobian_evaluator(solver.vec_jacobian),static,problem.get_variable_parameter_number()
    original_capture = SmallSignalStabilityEmtDriver._capture_limit_cycle_and_evaluator
    SmallSignalStabilityEmtDriver._capture_limit_cycle_and_evaluator = exact_capture
    try:
        driver.run()
    finally:
        SmallSignalStabilityEmtDriver._capture_limit_cycle_and_evaluator = original_capture
    metrics['exact_orbit_multipliers'] = [[float(z.real),float(z.imag)] for z in driver.results.multipliers]
    metrics['exact_orbit_max_magnitude'] = float(np.max(np.abs(driver.results.multipliers)))
    metrics['elapsed_s'] = perf_counter()-started
    output = Path(__file__).with_name('ieee39_initialization_audit.json')
    output.write_text(json.dumps(metrics,indent=2)+'\n')
    print('FINAL_AUDIT', json.dumps(metrics), flush=True)

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
