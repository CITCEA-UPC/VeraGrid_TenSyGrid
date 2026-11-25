"""Capture native converter control states and every procedural PWM boundary."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from TensyGridEngine.emt.validate_converter_emt_multilinear import build_converter_case, build_emt_options
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.solvers.structural_compiled_solver import StructuralCompiledWarmupPolicy
from VeraGridEngine.Templates.Emt.converter_switched_emt_template import get_switched_emt_converter
from VeraGridEngine.Utils.procedural_logic import build_boundary_updater_from_block, ThreePhaseCarrierPwmLogic
from VeraGridEngine.enumerations import EmtSolverTypes, DynamicIntegrationMethod
from VeraGridEngine.Utils.Symbolic.symbolic import Const


def build_filter_experiment(vf, name, current_filter_periods=5.):
    """Change only the three current-measurement filter rates in this experiment."""
    template = get_switched_emt_converter(vf, name)
    changed = []
    for block in template.block.get_all_blocks():
        for row, state in enumerate(block.state_vars):
            if state.name in ('i_d_meas_f', 'i_q_meas_f', 'i_0_meas_f'):
                block.state_eqs[row] = block.state_eqs[row] * Const(5./current_filter_periods)
                changed.append(state.name)
    if sorted(changed) != ['i_0_meas_f', 'i_d_meas_f', 'i_q_meas_f']:
        raise RuntimeError(f'Unexpected filter experiment targets: {changed}')
    return template


class RecordingUpdater:
    """Observe the production prepare/commit API without modifying gates or factors."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.pwm = next(entry for entry in delegate.logic_entries if isinstance(entry, ThreePhaseCarrierPwmLogic))
        self.rows = []
        self.accepted_times = []
        self.accepted_states = []
        self.last_ms = -1

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def get_next_forced_event_time(self, t_prev, t_target):
        return self.delegate.get_next_forced_event_time(t_prev, t_target)

    def prepare_interval(self, t, x, params):
        return self.delegate.prepare_interval(t, x, params)

    def resolve_step(self, accepted, params):
        return self.delegate.resolve_step(accepted, params)

    def commit_boundary(self, t, x, params):
        self.accepted_times.append(float(t))
        self.accepted_states.append(x.copy())
        result = self.delegate.commit_boundary(t, x, params)
        self.record_endpoint(t, x, params)
        return result

    def update(self, t, x, params):
        if not self.accepted_times and abs(t) < 1e-15:
            self.accepted_times.append(float(t))
            self.accepted_states.append(x.copy())
        result = self.delegate.update(t, x, params)
        self.record_endpoint(t, x, params)
        return result

    def record_endpoint(self, t, x, params):
        p = self.pwm
        self.rows.append([t, float(params[p.omega_sw_idx]), float(params[p.carrier_phase_idx]),
                          p.interval_end_time, *p.pending_transition_time,
                          *p.current_gate, x[p.mod_a_idx], x[p.mod_b_idx], x[p.mod_c_idx]])
        if int((t+1e-12)*1000) > self.last_ms:
            self.last_ms = int((t+1e-12)*1000)
            print(f'Boundary reached {t*1000:.3f} ms', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration-ms', type=float, default=10.)
    parser.add_argument('--step-us', type=float, default=10.)
    parser.add_argument('--current-filter-periods', type=float, default=5.,
                        help='Current feedback time constant in carrier periods; baseline is 5 (1 ms).')
    args = parser.parse_args()
    if args.current_filter_periods <= 0:
        parser.error('current-filter-periods must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    options = build_emt_options()
    options.solver_type = EmtSolverTypes.StructuralCompiled
    options.compiled_warmup_policy = StructuralCompiledWarmupPolicy.Off
    options.integration_method = DynamicIntegrationMethod.DaeBackEuler
    options.init_newton_max_iter = 60
    options.time_step = args.step_us/1e6
    options.simulation_time = args.duration_ms/1e3
    def builder(vf, name):
        return build_filter_experiment(vf, name, args.current_filter_periods)
    problem, _ = build_converter_case('native_control_diagnostic', builder, options)
    solver = build_emt_solver(options=options, problem=problem, t0=0., t_end=options.simulation_time,
                              h=options.time_step, method=options.integration_method)
    recorder = RecordingUpdater(build_boundary_updater_from_block(problem))
    setup = perf_counter()-started
    started = perf_counter()
    t, x, dx, initialized, converged = solver.simulate(boundary_updater=recorder)
    variables = problem.state_and_algebraic_vars()
    metadata = dict(initialized=bool(initialized), converged=bool(converged),
                    initialization_residual=float(problem.initialization_report.final_residual_inf),
                    setup_seconds=setup, solve_seconds=perf_counter()-started,
                    current_filter_periods=args.current_filter_periods,
                    current_filter_seconds_at_5khz=args.current_filter_periods/5000.,
                    nominal_step_seconds=options.time_step,
                    defer_pwm_edges=True, production_boundary_logic=True,
                    factorization_policy='solver-managed step and parameter snapshots',
                    edge_output_convention='left limit at PWM edges',
                    variables=[dict(name=v.name, uid=v.uid, index=int(problem.get_var_idx(v))) for v in variables],
                    pwm_columns=['time', 'omega', 'phase', 'interval_end', 'pending_a', 'pending_b', 'pending_c',
                                 'gate_a', 'gate_b', 'gate_c', 'm_a_input', 'm_b_input', 'm_c_input'])
    if not converged:
        residual = solver._residual_buffer
        equations = problem.get_state_eqs()+problem.get_algebraic_eqs()
        worst = np.argsort(abs(residual))[-5:][::-1]
        metadata['failure_residuals'] = [dict(row=int(row), residual=float(residual[row]),
                                             equation=str(equations[row])) for row in worst]
    np.savez_compressed(args.output/'controls.npz', time=t, x=x, dx=dx, pwm=np.asarray(recorder.rows),
                        accepted_time=np.asarray(recorder.accepted_times),
                        accepted_x=np.asarray(recorder.accepted_states))
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('variables', 'pwm_columns')}), flush=True)


if __name__ == '__main__':
    main()
