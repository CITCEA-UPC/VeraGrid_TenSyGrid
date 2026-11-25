"""Run the same converter grid against one explicitly selected engine source tree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--initialization', default='Explicit', choices=['Explicit', 'Auto', 'ConsistentNewton'])
parser.add_argument('--duration', type=float, default=.002)
parser.add_argument('--backend', choices=['Symbolic', 'StructuralCompiled'], default='Symbolic')
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block
from VeraGridEngine.Utils.procedural_logic import build_boundary_updater_from_block
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes, SolverType,
    VarPowerFlowReferenceType,
)
try:
    from VeraGridEngine.Templates.Emt.buses.bus_emt_template import get_bus_emt_template
    from VeraGridEngine.Templates.Emt.converters.converter_switched_emt_template import get_switched_emt_converter
    from VeraGridEngine.Templates.Emt.dc_line.dc_load_emt_template import get_dc_load_emt_template
    from VeraGridEngine.Templates.Emt.transformers.transformer_emt_template import get_transformer_emt_template
    from VeraGridEngine.Templates.Emt.synchronous_machines.thevenin_equivalent_emt_generator_template import get_generator_thevenin_rl_emt_template_with_ref
except ModuleNotFoundError:
    from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
    from VeraGridEngine.Templates.Emt.converter_switched_emt_template import get_switched_emt_converter
    from VeraGridEngine.Templates.Emt.dc_load_emt_template import get_dc_load_emt_template
    from VeraGridEngine.Templates.Emt.transformer_emt_template import get_transformer_emt_template
    from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import get_generator_thevenin_rl_emt_template_with_ref


def run() -> None:
    """Build, initialize and simulate the three-bus benchmark without patching the engine."""
    assert Path(gce.__file__).resolve().is_relative_to(args.source.resolve()), gce.__file__
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    grid = gce.MultiCircuit(name='Upstream initialization probe', Sbase=100., fbase=50.)
    slack = gce.Bus(name='Bus_Slack_AC', Vnom=230., is_slack=True)
    ac = gce.Bus(name='Bus_AC', Vnom=230.)
    dc = gce.Bus(name='Bus_DC', Vnom=320., is_dc=True)
    gen = gce.Generator(name='Gen_Thev', vset=1., Snom=100., freq=50., r1=.001, x1=.04)
    trafo = gce.Transformer2W(name='Trafo_AC', bus_from=slack, bus_to=ac, rate=100.,
                            r=.01, x=.1, tap_module=1., tap_phase=0.)
    vsc = gce.VSC(name='vsc_cmp', bus_from=dc, bus_to=ac, rate=100.,
                  control1=gce.ConverterControlType.Qac, control2=gce.ConverterControlType.Vm_dc,
                  control1_val=0., control2_val=1.)
    load = gce.Load(name='DC_Load', P=30., Q=0.)
    for bus in (slack, ac, dc):
        grid.add_bus(bus)
    grid.add_generator(slack, gen)
    grid.add_transformer2w(trafo)
    grid.add_vsc(vsc)
    grid.add_load(dc, load)
    grid.add_emt_events_group(gce.EmtEventsGroup(name='init_probe'))
    for bus in grid.buses:
        get_bus_emt_template(grid, bus)
    pf = gce.power_flow(grid=grid, options=PowerFlowOptions(
        solver_type=SolverType.NR, retry_with_other_methods=False, verbose=0,
        max_iter=30, tolerance=1e-6, control_q=False, control_taps_modules=False,
        control_taps_phase=False, orthogonalize_controls=False))
    if not pf.converged:
        raise RuntimeError('Power flow did not converge')
    model = get_switched_emt_converter(vf=grid.var_factory, name=vsc.name).block
    set_emt_model(gen, get_generator_thevenin_rl_emt_template_with_ref(vf=grid.var_factory).block, grid.var_factory)
    set_emt_model(trafo, get_transformer_emt_template(vf=grid.var_factory, name=trafo.name).block, grid.var_factory)
    set_emt_model(vsc, model, grid.var_factory)
    set_emt_model(load, get_dc_load_emt_template(vf=grid.var_factory, name='dc_load_emt_cmp').block, grid.var_factory)
    print(f'Templates built: {perf_counter() - started:.3f} s', flush=True)
    opt = EmtOptions(time_step=1e-5, simulation_time=args.duration, tolerance=1e-6,
                     solver_type=EmtSolverTypes[args.backend],
                     integration_method=DynamicIntegrationMethod.DaeBackEuler,
                     initialization_method=EmtInitializationMethod[args.initialization], verbose=0)
    from VeraGridEngine.Simulations.EMT.solvers.structural_compiled_solver import StructuralCompiledWarmupPolicy
    opt.compiled_warmup_policy = StructuralCompiledWarmupPolicy.Off
    opt.init_newton_max_iter = 60
    problem = EmtProblemDae(grid=grid, options=opt, pf_results=pf, pf_results_3ph=None)
    solver = build_emt_solver(options=opt, problem=problem, t0=0., t_end=args.duration,
                              h=1e-5, method=opt.integration_method)
    seed = problem.get_x0().copy()
    seed_dx = problem.get_dx0().copy()
    print(f'Problem/solver built: {perf_counter() - started:.3f} s', flush=True)
    setup_s = perf_counter() - started
    started = perf_counter()
    time, values, derivatives, initialized, converged = solver.simulate(
        boundary_updater=build_boundary_updater_from_block(problem))
    elapsed = perf_counter() - started
    signals = dict(time_arr=time)
    initial = dict()
    for key, reference in (('i_a', VarPowerFlowReferenceType.i_A),
                           ('i_b', VarPowerFlowReferenceType.i_B),
                           ('i_c', VarPowerFlowReferenceType.i_C),
                           ('i_dc', VarPowerFlowReferenceType.Idc),
                           ('p', VarPowerFlowReferenceType.P), ('q', VarPowerFlowReferenceType.Q),
                           ('v_dc_bus', VarPowerFlowReferenceType.Vdc)):
        idx = problem.get_var_idx(model.external_mapping[reference])
        signals[key] = values[:, idx]
        initial[key] = float(seed[idx])
    for name in ('v_dc', 'v_conv_a', 'gate_a', 'i_d', 'i_q', 'i_d_ref', 'i_q_ref',
                 'v_cmd_d', 'v_cmd_q', 'v_d', 'v_q', 'theta_pll', 'P_f', 'Q_f', 'switching_blend'):
        variable = find_name_in_block(name, model)
        if variable is not None:
            try:
                idx = problem.get_var_idx(variable)
                signals[name] = values[:, idx]
                initial[name] = float(seed[idx])
            except KeyError:
                pass
    np.savez_compressed(args.output / 'trajectory.npz', **signals)
    np.savez_compressed(args.output / 'initial_state.npz', x0=seed, dx0=seed_dx,
                        names=np.array([v.name for v in problem.state_and_algebraic_vars()]))
    report = problem.initialization_report
    summary = dict(engine_file=gce.__file__, initialization=args.initialization, backend=args.backend,
                   initialized=bool(initialized), converged=bool(converged),
                   setup_seconds=setup_s, simulation_seconds=elapsed,
                   final_time=float(time[-1]), points=len(time), initial=initial,
                   first_sample={k: float(v[0]) for k, v in signals.items()},
                   last_sample={k: float(v[-1]) for k, v in signals.items()},
                   initialization_report=dict(method=str(report.method_used), status=str(report.status),
                       message=report.message, initial_residual_inf=report.initial_residual_inf,
                       final_residual_inf=report.final_residual_inf,
                       automatic_dx0_count=report.automatic_dx0_count))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    run()
