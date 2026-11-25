"""Build/export and simulate compact multilinear IEEE9 and Kundur demos.

Run from the repository root: PYTHONPATH=src:. python -m demo_consortium.run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import VeraGridEngine.api as gce
from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Devices.Events.emt_events_group import EmtEventsGroup
from VeraGridEngine.Devices.Events.rms_event import RmsEvent
from VeraGridEngine.Devices.Events.rms_events_group import RmsEventsGroup
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.EMT.solvers.structural_compiled_solver import StructuralCompiledWarmupPolicy
from VeraGridEngine.Simulations.Rms.problems.rms_problem_phasor import RmsProblemPhasor
from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
from VeraGridEngine.Simulations.Rms.numerical.trapezoidal import TrapezoidalImplicitIntegration
from VeraGridEngine.enumerations import EmtInitializationMethod, EmtSolverTypes, DynamicIntegrationMethod, RmsInitializationMethod
from demo_consortium.cases import build_case


def make_problem(grid, pf, emt: bool, duration: float, step: float):
    """Build the native engine problem; multilinear device equations are preserved."""
    if emt:
        options = EmtOptions(time_step=step, simulation_time=duration, tolerance=1e-7,
            solver_type=EmtSolverTypes.StructuralCompiled, initialization_method=EmtInitializationMethod.Explicit,
            integration_method=DynamicIntegrationMethod.DaeTrapezoidal, verbose=0)
        options.compiled_warmup_policy = StructuralCompiledWarmupPolicy.Off
        options.init_newton_max_iter = 50
        return EmtProblemDae(grid=grid, options=options, pf_results=pf, pf_results_3ph=None)
    options = RmsOptions(time_step=step, simulation_time=duration, tolerance=1e-8,
                         initialization_method=RmsInitializationMethod.Explicit, verbose=0)
    return RmsProblemPhasor(grid=grid, options=options, pf_results=pf)


def integrate(problem, emt: bool, duration: float, step: float):
    """Use native trapezoidal integration for smooth averaged EMT and RMS models."""
    if emt:
        solver = build_emt_solver(problem.options, problem, 0., duration, step,
                                 method=DynamicIntegrationMethod.DaeTrapezoidal)
        t, x, _, initialized, converged = solver.simulate()
    else:
        solver = TrapezoidalImplicitIntegration(problem, 0., duration, step, 50, 1e-8,
                                                use_fd_jacobian=False)
        t, x, initialized, converged = solver.simulate()
    return t, x, bool(initialized), bool(converged)


def initial_residual(problem, emt: bool) -> float:
    """Evaluate the actual initial algebraic equations independently of solver flags."""
    x, dx = problem.get_x0(), problem.get_dx0()
    runtime = problem.event_params_values if emt else problem._variable_parameters_values
    parameters = problem.get_parameters_values()
    values = {uid: float(x[index]) for uid, index in problem._uid2idx_vars.items()}
    values.update({uid: float(dx[index]) for uid, index in problem._uid2idx_diff.items()})
    values.update({uid: float(runtime[index]) for uid, index in problem._uid2idx_event_params.items()})
    values.update({uid: float(parameters[index].value) for uid, index in problem._uid2idx_params.items()})
    if emt:
        values[problem.glob_time.uid] = 0.
    equations = problem.get_algebraic_eqs() if emt else problem.get_algebraic_eqs
    return max(abs(float(eq.eval_uid(values))) for eq in equations)


def add_load_event(grid, pf, emt: bool, time: float, scale: float):
    """Increase the first load's active conductance; leave its reactive element unchanged."""
    load = grid.loads[0]
    if emt:
        group = EmtEventsGroup(name='2 percent active load increase')
        grid.add_emt_events_group(group)
        index = list(grid.buses).index(load.bus)
        resistance = abs(pf.voltage[index])**2/(load.P/grid.Sbase)
        for owner in load.emt_model.get_all_blocks():
            for parameter in owner.event_dict:
                if parameter.name in ('R_A', 'R_B', 'R_C'):
                    grid.add_emt_event(EmtEvent(name=f'{load.name} {parameter.name} step', device=load,
                        parameter=parameter, time=time, value=resistance/scale, group=group,
                        force_step_alignment=True))
    else:
        group = RmsEventsGroup(name='2 percent active load increase')
        grid.add_rms_events_group(group)
        parameter = next(v for v in load.rms_model.event_dict if v.name.endswith('.scale'))
        grid.add_rms_event(RmsEvent(name='Load conductance step', device=load, parameter=parameter,
                                   time=time, value=scale, group=group, force_step_alignment=True))
    return group


def extract_signals(problem, grid, x, emt: bool):
    """Extract named source signals from each device's declared variables."""
    signals = {}
    for generator in grid.generators:
        model = generator.emt_model if emt else generator.rms_model
        prefix = generator.name.replace(' ', '_')
        all_vars = [v for b in model.get_all_blocks() for v in [*b.state_vars, *b.algebraic_vars]]
        by_name = {v.name: v for v in all_vars}
        for quantity in ('omega', 'P', 'Q', 'E', 'p_limited', 'cos', 'sin'):
            variable = by_name.get(f'{prefix}.{quantity}')
            if variable is not None:
                signals[f'{generator.name}:{quantity}'] = x[:, problem.get_var_idx(variable)]
    return signals


def plot_signals(t, signals, grid, output: Path, event_time: float, title: str):
    """Plot frequency and system-base powers without confusing EMT ripple with RMS."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    for generator in grid.generators:
        for axis, quantity in zip(axes, ('omega', 'P', 'Q')):
            values = signals[f'{generator.name}:{quantity}']
            values = grid.fBase*(values-1.) if quantity == 'omega' else values*grid.Sbase
            axis.plot(t, values, label=generator.name, linewidth=.9)
    for axis, label in zip(axes, ('Frequency deviation [Hz]', 'Terminal active power [MW]', 'Terminal reactive power [MVAr]')):
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
        axis.axvline(event_time, color='gray', linestyle=':', linewidth=1.)
        axis.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel('Time [s]')
    axes[0].set_title(title)
    fig.savefig(output/'response.png', dpi=150)
    plt.close(fig)


def run_case(case: str, emt: bool, duration: float, step: float, output_root: Path):
    """Export the model-bearing grid and validate a complete load-step trajectory."""
    started = perf_counter()
    output = output_root/case/('emt' if emt else 'rms')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('VERAGRID_EMT_INIT_CACHE_DIR', str(output_root/'.cache'))
    grid, pf, inverter = build_case(case, emt)
    event_time = .05
    if duration <= event_time:
        raise ValueError('Duration must exceed the 0.05 s load event')
    group = add_load_event(grid, pf, emt, event_time, 1.02)
    # The files contain the actual attached symbolic device models and event definitions.
    gce.save_file(grid, str(output/'grid.veragrid'))
    problem = make_problem(grid, pf, emt, duration, step)
    problem.set_events_group(group)
    residual = initial_residual(problem, emt)
    setup = perf_counter()-started
    started = perf_counter()
    t, x, initialized, converged = integrate(problem, emt, duration, step)
    signals = extract_signals(problem, grid, x, emt)
    finite = bool(np.isfinite(x).all())
    valid = initialized and converged and finite and t[-1] >= duration-1e-10 and residual < 1e-6
    summary = dict(case=case, formulation='EMT' if emt else 'RMS', model='lifted classical SG / averaged VSM GFM',
        buses=len(grid.buses), synchronous_generators=len(grid.generators)-1, inverters=1,
        inverter=inverter.name, nominal_frequency_hz=grid.fBase, system_base_mva=grid.Sbase,
        power_flow_converged=bool(pf.converged), initialized=initialized, converged=converged,
        finite=finite, initial_algebraic_residual=residual, final_time=float(t[-1]),
        step_seconds=step, requested_duration_seconds=duration, event_time=event_time,
        event='first load active conductance +2%', samples=len(t), variables=x.shape[1],
        setup_seconds=setup, solve_seconds=perf_counter()-started, valid=valid)
    for generator in grid.generators:
        c, s = signals[f'{generator.name}:cos'], signals[f'{generator.name}:sin']
        summary[f'{generator.name}:max_trig_manifold_error'] = float(np.max(abs(c*c+s*s-1)))
    np.savez_compressed(output/'traces.npz', time=t, **signals)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot_signals(t, signals, grid, output, event_time, grid.name)
    print(json.dumps(summary), flush=True)
    if not valid:
        raise RuntimeError(f'{case} {summary["formulation"]} failed validation; see {output}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['ieee9', 'kundur11', 'all'], default='all')
    parser.add_argument('--formulation', choices=['emt', 'rms', 'all'], default='all')
    parser.add_argument('--emt-duration', type=float, default=.2)
    parser.add_argument('--rms-duration', type=float, default=2.)
    parser.add_argument('--emt-step', type=float, default=.0001)
    parser.add_argument('--rms-step', type=float, default=.002)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    for case in (['ieee9', 'kundur11'] if args.case == 'all' else [args.case]):
        for formulation in (['rms', 'emt'] if args.formulation == 'all' else [args.formulation]):
            emt = formulation == 'emt'
            run_case(case, emt, args.emt_duration if emt else args.rms_duration,
                     args.emt_step if emt else args.rms_step, args.output.resolve())


if __name__ == '__main__':
    main()
