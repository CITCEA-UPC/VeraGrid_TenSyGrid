#!/usr/bin/env python3
"""IEEE39/IEEE118 RMS and native/multilinear EMT SSA benchmarks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import platform
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
for key in ('NUMBA_CACHE_DIR', 'MPLCONFIGDIR', 'VERAGRID_JIT_CACHE_DIR', 'VERAGRID_EMT_INIT_CACHE_DIR'):
    os.environ.setdefault(key, '/tmp/veragrid-benchmark-' + key.lower())
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('VERAGRID_RUNTIME_PARAMETER_PYTHON', '1')
GRIDS = {
    'ieee39': ROOT / 'Grids_and_profiles/grids/IEEE39.gridcal',
    'ieee118': ROOT / 'src/tests/data/grids/RAW/IEEE 118 Bus.raw',
}


def attach_rms(grid):
    import VeraGridEngine.api as vge
    from VeraGridEngine.Templates.Rms.transformer_rms_template_legacy import initialize_trafo_rms
    from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_rms_model
    from VeraGridEngine.Templates.Rms.genqec_phasor_rms_template import get_genqec_phasor
    from VeraGridEngine.Utils.Symbolic.block import find_name_in_block
    for bus in grid.buses:
        vge.initialize_bus_phasor_rms(bus, grid.var_factory)

    def attach(device, model, buses, implicit=True):
        inputs = [v for bus in buses for v in bus.rms_model.out_vars[:2]]
        grid.var_factory.add_connections(model.in_vars[:len(inputs)], inputs)
        if implicit:
            model = vge.to_implicit(model, grid.var_factory)
        set_rms_model(device, model, grid.var_factory)

    for i, gen in enumerate(grid.generators):
        model = get_genqec_phasor(grid.var_factory, name=f'generator_{i}').block
        # Hold mechanical torque and field voltage at their PF-seeded values.
        model.event_dict[model.in_vars[2]] = find_name_in_block('Te', model)
        model.event_dict[model.in_vars[3]] = find_name_in_block('IRPu', model)
        attach(gen, model, [gen.bus])
    for i, line in enumerate(grid.lines):
        attach(line, vge.get_line_phasor_rms_template(grid.var_factory, name=f'line_{i}').block,
               [line.bus_from, line.bus_to])
    for i, load in enumerate(grid.loads):
        attach(load, vge.get_load_phasor_current_rms_template(
            grid.var_factory, name=f'load_{i}').block, [load.bus], implicit=False)
    for trafo in grid.transformers2w:
        attach(trafo, initialize_trafo_rms(trafo, grid.var_factory, Sbase=grid.Sbase,
                                         use_phasor_template=True).block,
               [trafo.bus_from, trafo.bus_to])
    for i, shunt in enumerate(grid.shunts):
        attach(shunt, vge.get_shunt_template(grid.var_factory, name=f'shunt_{i}', phasor=True).block,
               [shunt.bus])


def attach_emt(grid, transformer_model="coupled_core", emt_models="auxiliary"):
    from VeraGridEngine.Templates.Emt.simple_generator_emt_multilinear_template import get_simple_generator_emt_multilinear_template
    from VeraGridEngine.Templates.Emt.load_zip_emt_multilinear_template import get_load_ZIP_emt_multilinear_template
    from VeraGridEngine.Templates.Emt.xfmr_emt_multilinear_template import get_xfmr_emt_template_multilinear
    from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
    from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
    from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
    from VeraGridEngine.Utils.Symbolic.block import find_name_in_block
    if emt_models == 'canonical':
        from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_simple_generator_emt_template
        from VeraGridEngine.Templates.Emt.load_zip_emt_template import get_load_ZIP_emt_template
        from VeraGridEngine.Templates.Emt.xfmr_emt_template import get_xfmr_emt_template
        generator_factory = get_simple_generator_emt_template
        load_factory = get_load_ZIP_emt_template
    else:
        generator_factory = get_simple_generator_emt_multilinear_template
        load_factory = get_load_ZIP_emt_multilinear_template
    for bus in grid.buses:
        get_bus_emt_template(grid, bus)
    for i, line in enumerate(grid.lines):
        set_emt_model(line, get_pi_line_emt_template(grid.var_factory, name=f'line_{i}').block,
                      grid.var_factory)
    # Static benchmark cases omit machine sequence data. Supply explicit generic
    # assumptions for dynamic/PF3 models; retain the positive-sequence dispatch.
    for gen in grid.generators:
        for resistance, reactance, default_x in (('R1', 'X1', 1.8), ('R0', 'X0', .15), ('R2', 'X2', .2)):
            if abs(complex(getattr(gen, resistance), getattr(gen, reactance))) < 1e-12:
                setattr(gen, resistance, .003)
                setattr(gen, reactance, default_x)
    for i, gen in enumerate(grid.generators):
        model = generator_factory(grid.var_factory, name=f'generator_{i}').block
        torque_name = 'Te' if emt_models == 'canonical' else f'Te_generator_{i}'
        model.event_dict[model.in_vars[3]] = find_name_in_block(torque_name, model)
        model.event_dict[model.in_vars[4]] = (next(v for v in model.event_dict if v.name == 'Rf')
                                                * find_name_in_block('i_f', model))
        set_emt_model(gen, model, grid.var_factory)
    for kind, devices in (('load', grid.loads), ('shunt', grid.shunts)):
        for i, device in enumerate(devices):
            model = load_factory(grid.var_factory, name=f'{kind}_{i}').block
            coefficients = (0, 0, 1, 0, 0, 1) if kind == 'load' else (1, 0, 0, 1, 0, 0)
            for index, value in enumerate(coefficients, 1):
                var = next((v for v in model.event_dict if v.name == f'a{index}'), None)
                if var is None:
                    raise KeyError(f'Missing ZIP coefficient a{index}')
                model.event_dict[var] = grid.var_factory.add_const(float(value))
            if kind == 'shunt':
                for block in model.get_all_blocks():
                    for reference, var in list(block.api_obj_mapping.items()):
                        if var.name.startswith(('P0_', 'Q0_')):
                            value = device.G if var.name.startswith('P0_') else -device.B
                            block.event_dict[var] = grid.var_factory.add_const(float(value / grid.Sbase / 3))
                            del block.api_obj_mapping[reference]
            set_emt_model(device, model, grid.var_factory)
    for i, trafo in enumerate(grid.transformers2w):
        if transformer_model == 'simple_linear':
            from trunk.benchmarks.transformer_equivalence import assert_simple_compatible, get_simple_linear_transformer
            assert_simple_compatible(trafo)
            model = get_simple_linear_transformer(grid.var_factory, name=f'transformer_{i}').block
        elif emt_models == 'canonical':
            model = get_xfmr_emt_template(grid.var_factory, conn_f=trafo.conn_f,
                conn_t=trafo.conn_t, name=f'transformer_{i}').block
        else:
            model = get_xfmr_emt_template_multilinear(grid.var_factory, name=f'transformer_{i}').block
        set_emt_model(trafo, model, grid.var_factory)


def periodicity_report(trajectory, times, names, angle_indices, period, tolerance):
    """Measure one-cycle closure, including the winding of absolute angles."""
    import numpy as np
    if len(trajectory) < 2 or len(times) != len(trajectory):
        raise ValueError('A complete trajectory and its times are required')
    closure = trajectory[-1] - trajectory[0]
    closure = closure.copy()
    closure[angle_indices] -= 2 * np.pi
    scales = np.maximum(1.0, np.max(np.abs(trajectory), axis=0))
    errors = np.abs(closure) / scales
    duration = float(times[-1] - times[0])
    duration_valid = bool(np.isclose(duration, period, rtol=1e-9, atol=1e-12))
    largest = np.argsort(errors)[-10:][::-1]
    return dict(last_period_scaled_closure_inf=float(np.max(errors)),
                periodicity_tolerance=tolerance, captured_period_seconds=duration,
                target_period_seconds=period, absolute_angle_count=len(angle_indices),
                periodicity_validated=bool(duration_valid and np.all(np.isfinite(errors))
                                          and np.max(errors) <= tolerance),
                largest_period_closure_errors=[dict(index=int(i), name=names[i],
                    difference=float(closure[i]), scaled_error=float(errors[i])) for i in largest])


def source_fingerprint(directory):
    """Hash Python source contents and relative paths, independent of checkout path."""
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob('*.py')):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def checkpoint(args, record):
    """Keep completed phase timings available if a child is timed out."""
    (args.output / f'{args.grid}_{args.formulation}.json').write_text(
        json.dumps(record, indent=2) + '\n')


def run_case(args, record):
    import numpy as np
    import VeraGridEngine.api as vge
    from VeraGridEngine.Simulations.Rms.problems.rms_problem_multilinear import RmsProblemMultilinear
    from VeraGridEngine.Simulations.EMT.problems.emt_problem_multilinear import EmtProblemMultilinear
    from VeraGridEngine.Simulations.SmallSignalStabilityRms.small_signal_driver import run_dense_small_signal_stability, run_sparse_small_signal_stability
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import SmallSignalStabilityEmtDriver
    from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import SmallSignalStabilityEmtOptions
    from VeraGridEngine.enumerations import (
        EmtProblemTypes, SmallSignalEmtBuildTypes, RmsProblemTypes, RmsInitializationMethod, EmtSolverTypes,
        DynamicIntegrationMethod, EmtInitializationMethod)
    path = Path(args.grid_file).resolve() if args.grid_file else GRIDS[args.grid]
    record.update(grid_file=str(path), formulation=args.formulation, stage='load')
    record['grid_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    record['python_version'] = platform.python_version()
    record['numpy_version'] = np.__version__
    import VeraGridEngine
    record['engine_module'] = VeraGridEngine.__file__
    record['engine_source_sha256'] = source_fingerprint(Path(VeraGridEngine.__file__).parent)
    record['benchmark_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record['transformer_helper_sha256'] = hashlib.sha256(
        Path(__file__).with_name('transformer_equivalence.py').read_bytes()).hexdigest()
    record['machine_profile'] = 'template defaults; fixed PF-seeded torque and field voltage'
    print(f'Loading {args.grid} {args.formulation}: {path}', flush=True)
    start_load = time.perf_counter()
    grid = vge.open_file(str(path))
    record['load_seconds'] = time.perf_counter() - start_load
    record['buses'] = len(grid.buses)
    record['generators'] = len(grid.generators)
    record['transformers'] = len(grid.transformers2w)
    record['transformer_model'] = args.transformer_model
    record['emt_models'] = args.emt_models
    record['stage'] = 'attach_models'
    start_attach = time.perf_counter()
    if not args.reuse_models:
        if args.formulation == 'rms':
            attach_rms(grid)
        else:
            attach_emt(grid, args.transformer_model, args.emt_models)
    record['attach_seconds'] = time.perf_counter() - start_attach
    record['stage'] = 'power_flow'
    pf = vge.PowerFlowDriver(grid, vge.PowerFlowOptions(tolerance=1e-8))
    start_pf = time.perf_counter()
    pf.run()
    record['balanced_pf_seconds'] = time.perf_counter() - start_pf
    record['balanced_pf_converged'] = bool(pf.results.converged)
    if not pf.results.converged:
        raise RuntimeError('Balanced power flow failed')
    print('Balanced PF converged; building dynamic problem', flush=True)
    record['stage'] = 'build_problem'
    start = time.perf_counter()
    if args.formulation == 'rms':
        options = vge.RmsOptions(problem_type=RmsProblemTypes.Multilinear,
            initialization_method=(RmsInitializationMethod.ConsistentNewton
                if args.rms_init == 'consistent' else RmsInitializationMethod.Explicit), max_iter=50)
        problem = RmsProblemMultilinear(grid=grid, options=options, pf_results=pf.results)
        record.update(problem_class=type(problem).__name__, states=problem.get_states_number(),
                      algebraics=problem.get_algebraic_var_number(),
                      derivatives=problem.get_diff_var_number())
        record['build_seconds'] = time.perf_counter() - start
        x0 = problem.get_x0()
        dx0 = np.zeros_like(x0)
        matrix_start = time.perf_counter()
        problem.build_multilinear_matrices()
        record['matrix_build_seconds'] = time.perf_counter() - matrix_start
        residual = problem.exact_residual_from_s_phi(
            np.concatenate((x0, np.zeros(problem.get_diff_var_number()))))
        record['initial_residual_inf'] = float(np.max(np.abs(residual)))
        rows = np.argsort(np.abs(residual))[-5:][::-1]
        record['largest_initial_residual_rows'] = [dict(row=int(i), residual=float(residual[i])) for i in rows]
        record['rms_descriptor_source_sha256'] = hashlib.sha256(
            Path(__file__).with_name('rms_descriptor.py').read_bytes()).hexdigest()
        record['stage'] = 'representation_validation'
        from trunk.benchmarks.rms_descriptor import check_symbolic_representation
        record.update(check_symbolic_representation(problem, x0))
        initialized_voltage = np.array([
            x0[problem.get_var_idx(bus.rms_model.out_vars[0])]
            + 1j*x0[problem.get_var_idx(bus.rms_model.out_vars[1])] for bus in grid.buses])
        record['power_flow_voltage_difference_inf'] = float(np.max(abs(initialized_voltage - pf.results.voltage)))
        record['equilibrium_validated'] = bool(np.all(np.isfinite(residual))
                                              and record['initial_residual_inf'] <= options.tolerance)
        if not record['equilibrium_validated'] and not args.allow_nonequilibrium:
            record['stage'] = 'equilibrium_validation'
            raise RuntimeError('RMS initial residual exceeds tolerance; SSA was stopped. '
                               'Use --allow-nonequilibrium only for diagnostics.')
        print('Problem built; running SSA', flush=True)
        record['stage'] = 'ssa'
        start = time.perf_counter()
        checkpoint(args, record)
        if args.rms_spectrum == 'reduced':
            from trunk.benchmarks.rms_descriptor import finite_descriptor_modes
            values, _, descriptor_report = finite_descriptor_modes(problem, x0, dx0)
            record.update(descriptor_report)
        elif args.rms_spectrum == 'full':
            record['descriptor_regularization'] = 1e-7
            record['method'] = 'Legacy dense descriptor solver with diagonal regularization'
            values = run_dense_small_signal_stability(problem, x0, dx0)[0]
        else:
            values = run_sparse_small_signal_stability(problem, x0, dx0, k=args.modes)[0]
        multipliers = None
    else:
        from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
        from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
        pf3 = PowerFlowDriver3Ph(grid, vge.PowerFlowOptions(tolerance=1e-8))
        record['stage'] = 'three_phase_power_flow'
        checkpoint(args, record)
        pf3.run()
        record['three_phase_pf_seconds'] = time.perf_counter() - start
        record['three_phase_pf_converged'] = bool(pf3.results.converged)
        if not pf3.results.converged:
            raise RuntimeError('Three-phase power flow failed')
        record['stage'] = 'build_problem'
        checkpoint(args, record)
        print('Three-phase PF converged; constructing EMT DAE', flush=True)
        options = EmtOptions(tolerance=1e-6,
            integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
            initialization_method=EmtInitializationMethod.Explicit, verbose=0)
        options.problem_type = getattr(EmtProblemTypes, args.emt_problem)
        record['emt_problem_requested'] = args.emt_problem
        options.solver_type = getattr(EmtSolverTypes, args.emt_solver)
        period = 1 / grid.fBase
        options.time_step = period / args.steps_per_period
        options.simulation_time = period * args.assessment_periods
        ssa = SmallSignalStabilityEmtOptions(
            k=args.modes, target_period=period, ss_assessment_time=options.simulation_time,
            max_krylov_dim=max(args.krylov_dim, 2 * args.modes), max_restarts=1,
            build_type=SmallSignalEmtBuildTypes.Arnoldi)
        if args.consistent_tangent:
            if 'consistent_dae_tangent' not in type(ssa).__slots__:
                raise RuntimeError('This engine lacks consistent DAE tangents; use --no-consistent-tangent for its native SSA path')
            ssa.consistent_dae_tangent = True
        record['consistent_dae_tangent_requested'] = args.consistent_tangent
        start = time.perf_counter()
        class TimedEmtDriver(SmallSignalStabilityEmtDriver):
            def _capture_limit_cycle_and_evaluator(self, h, verbose=0):
                capture_start = time.perf_counter()
                try:
                    captured = super()._capture_limit_cycle_and_evaluator(h, verbose)
                    self.benchmark_trajectory = captured[0]
                    self.benchmark_derivatives = captured[1] if len(captured) == 6 else np.empty((0, 0))
                    if len(captured[0]) > 1:
                        from VeraGridEngine.Simulations.EMT.initialization_emt import _emt_is_absolute_angle
                        names = [''] * captured[0].shape[1]
                        angle_indices = []
                        multilinear_angles = {f'theta_generator_{i}' for i in range(len(grid.generators))}
                        for var in self.problem.get_state_vars() + self.problem.get_algebraic_vars():
                            index = self.problem.get_var_idx(var)
                            names[index] = var.name
                            if _emt_is_absolute_angle(var) or var.name in multilinear_angles:
                                angle_indices.append(index)
                        times = captured[2] if len(captured) == 6 else captured[1]
                        record.update(periodicity_report(captured[0], times, names, angle_indices,
                                                         period, args.periodicity_tolerance))
                        np.savez_compressed(args.output / f'{args.grid}_emt_trajectory.npz',
                            trajectory=captured[0], derivatives=self.benchmark_derivatives,
                            times=times, names=names, angle_indices=angle_indices,
                            bus_voltages=captured[0][:,[self.problem.get_var_idx(var)
                                for bus in grid.buses for var in bus.emt_model.out_vars]],
                            bus_names=[bus.name for bus in grid.buses])
                        checkpoint(args, record)
                        if not record['periodicity_validated'] and not args.allow_nonperiodic:
                            raise RuntimeError('Captured EMT trajectory is not periodic at the requested '
                                'tolerance; SSA was stopped. Use --allow-nonperiodic only for diagnostics.')
                    return captured
                finally:
                    record['trajectory_and_jacobian_setup_seconds'] = time.perf_counter() - capture_start
                    checkpoint(args, record)

        driver_arguments = dict(grid=grid, emt_options=options,
                                sss_options=ssa, pf_results=pf.results)
        if 'pf_results_3ph' in inspect.signature(SmallSignalStabilityEmtDriver.__init__).parameters:
            driver_arguments['pf_results_3ph'] = pf3.results
        driver = TimedEmtDriver(**driver_arguments)
        problem = driver.problem
        record.update(problem_class=type(problem).__name__, states=problem.get_states_number(),
                      algebraics=problem.get_algebraic_var_number(),
                      derivatives=problem.get_diff_var_number())
        record['build_seconds'] = time.perf_counter() - start
        record['emt_build_breakdown'] = problem.get_build_report()
        record['trajectory_solver_actual'] = args.emt_solver
        record['floquet_jacobian_actual'] = ('EmtProblemMultilinear S/Phi adapter'
            if isinstance(problem, EmtProblemMultilinear) else 'Native solver discrete Jacobian')
        report = problem.initialization_report
        record['initialization_report'] = dict(status=str(report.status),
            initial_residual_inf=float(report.initial_residual_inf),
            final_residual_inf=float(report.final_residual_inf), message=report.message)
        initial_residual = record['initialization_report']['final_residual_inf']
        record['initialization_validated'] = bool(np.isfinite(initial_residual)
                                                  and initial_residual <= options.tolerance)
        if not record['initialization_validated'] and not args.allow_nonequilibrium:
            record['stage'] = 'initialization_validation'
            checkpoint(args, record)
            raise RuntimeError('EMT initial DAE residual exceeds tolerance; capture and SSA were stopped. '
                               'Use --allow-nonequilibrium only for diagnostics.')
        print('Problem built; capturing EMT trajectory' if args.capture_only
              else 'Problem built; running SSA', flush=True)
        record['stage'] = 'capture' if args.capture_only else 'ssa'
        start = time.perf_counter()
        checkpoint(args, record)
        if args.capture_only:
            captured = driver._capture_limit_cycle_and_evaluator(options.time_step)
            if len(captured[0]) < 2:
                raise RuntimeError('EMT capture failed; no complete period is available')
            record.update(status='completed', stage='capture_only', modes=0,
                capture_only=True, capture_seconds=time.perf_counter()-start,
                interpretation='Trajectory comparison only; no stability modes were calculated.')
            return
        driver.run()
        record['floquet_after_capture_seconds'] = (time.perf_counter() - start
            - record.get('trajectory_and_jacobian_setup_seconds', 0.0))
        if driver.results is None:
            raise RuntimeError('EMT SSA returned no results; see capture diagnostics in the run log')
        values = driver.results.eigenvalues
        multipliers = driver.results.multipliers
        trajectory = driver.benchmark_trajectory
    record.update(ssa_seconds=time.perf_counter() - start,
                  problem_class=type(problem).__name__, states=problem.get_states_number(),
                  algebraics=problem.get_algebraic_var_number(),
                  derivatives=problem.get_diff_var_number())
    values = np.asarray(values, dtype=complex)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise RuntimeError('SSA returned no finite modes')
    with (args.output / f'{args.grid}_{args.formulation}_modes.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['real_per_s', 'imag_rad_per_s', 'frequency_hz', 'multiplier_real', 'multiplier_imag'])
        for i, val in enumerate(values):
            mu = complex(multipliers[i]) if multipliers is not None else None
            writer.writerow([val.real, val.imag, abs(val.imag)/(2*np.pi),
                             mu.real if mu is not None else '', mu.imag if mu is not None else ''])
    record['interpretation'] = ('Numerical SSA completed; operating-point or periodic-orbit validity '
                                'must be checked before interpreting stability.')
    record.update(status='completed', stage='finished', modes=len(values), finite_modes=len(finite),
                  max_real_per_s=float(finite.real.max()))
    if multipliers is not None:
        record['max_multiplier_magnitude'] = float(np.max(np.abs(multipliers)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', choices=GRIDS, default='ieee39')
    parser.add_argument('--engine-root', type=Path, help='Use another checkout source tree')
    parser.add_argument('--reuse-models', action='store_true', help='Use models saved in the input grid')
    parser.add_argument('--grid-file', help='Override the input case without altering its dispatch')
    parser.add_argument('--formulation', choices=['rms', 'emt'], default='rms')
    parser.add_argument('--consistent-tangent', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--emt-models', choices=['auxiliary', 'canonical'], default='auxiliary',
                        help='Original auxiliary models or standard non-lifted EMT templates')
    parser.add_argument('--emt-problem', choices=['CurrentBalance', 'Multilinear'], default='Multilinear',
                        help='Native DAE or S/Phi backend, with the same attached device equations')
    parser.add_argument('--emt-solver', choices=['StructuralAD', 'StructuralCompiled', 'Symbolic', 'Automatic'],
                        default='StructuralAD')
    parser.add_argument('--periodicity-tolerance', type=float, default=1e-4,
                        help='Maximum scaled one-period closure error before allowing EMT SSA')
    parser.add_argument('--allow-nonequilibrium', action='store_true',
                        help='Diagnostic only: allow an RMS/EMT initial residual above tolerance')
    parser.add_argument('--allow-nonperiodic', action='store_true',
                        help='Diagnostic only: compute modes even if one-cycle closure fails')
    parser.add_argument('--transformer-model', choices=['coupled_core','simple_linear'], default='coupled_core')
    parser.add_argument('--capture-only', action='store_true', help='Validate and save EMT capture without mode extraction')
    parser.add_argument('--all', action='store_true', help='Run all four combinations in isolated processes')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--timeout', type=float, default=1800, help='Seconds allowed per child in --all mode')
    parser.add_argument('--rms-init', choices=['explicit', 'consistent'], default='explicit')
    parser.add_argument('--rms-spectrum', choices=['sparse', 'full', 'reduced'], default='reduced',
                        help='Reduced computes all finite descriptor modes; sparse targets near zero; full uses the legacy dense solver')
    parser.add_argument('--krylov-dim', type=int, default=24, help='Arnoldi subspace dimension; increase if convergence is slow')
    parser.add_argument('--modes', type=int, default=6)
    parser.add_argument('--steps-per-period', type=int, default=600)
    parser.add_argument('--assessment-periods', type=int, default=1)
    args = parser.parse_args()
    if min(args.modes, args.krylov_dim, args.steps_per_period, args.assessment_periods, args.timeout) <= 0:
        parser.error('Mode count, steps, periods, and timeout must be positive')
    if not 0 < args.periodicity_tolerance < 1:
        parser.error('Periodicity tolerance must be between zero and one')
    if args.formulation == 'emt' and args.emt_models == 'canonical' and args.emt_problem != 'CurrentBalance':
        parser.error('Canonical EMT templates require --emt-problem CurrentBalance')
    if args.capture_only and (args.all or args.formulation != 'emt'):
        parser.error('--capture-only requires an individual EMT case')
    if args.all and args.grid_file:
        parser.error('--grid-file is only supported for individual runs')
    if args.engine_root:
        args.engine_root = args.engine_root.resolve()
        sys.path.insert(0, str(args.engine_root / 'src'))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.all:
        failed = False
        summary = []
        for grid in GRIDS:
            for formulation in ('rms', 'emt'):
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--grid', grid,
                           '--formulation', formulation, '--emt-models', args.emt_models, '--emt-problem', args.emt_problem, '--emt-solver', args.emt_solver, '--output', str(args.output), '--timeout', str(args.timeout),
                           '--modes', str(args.modes), '--krylov-dim', str(args.krylov_dim), '--rms-spectrum', args.rms_spectrum, '--rms-init', args.rms_init, '--steps-per-period', str(args.steps_per_period),
                           '--assessment-periods', str(args.assessment_periods),
                           '--periodicity-tolerance', str(args.periodicity_tolerance),
                           '--transformer-model', args.transformer_model]
                if args.capture_only:
                    command.append('--capture-only')
                if args.allow_nonperiodic:
                    command.append('--allow-nonperiodic')
                if args.allow_nonequilibrium:
                    command.append('--allow-nonequilibrium')
                if not args.consistent_tangent:
                    command.append('--no-consistent-tangent')
                if args.engine_root:
                    command.extend(['--engine-root', str(args.engine_root)])
                if args.reuse_models:
                    command.append('--reuse-models')
                (args.output / f'{grid}_{formulation}.json').unlink(missing_ok=True)
                print(f'Running {grid} {formulation}', flush=True)
                with (args.output / f'{grid}_{formulation}.log').open('w') as log:
                    try:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                        failed |= result.returncode != 0
                    except subprocess.TimeoutExpired:
                        failed = True
                        timed_path = args.output / f'{grid}_{formulation}.json'
                        timed_record = json.loads(timed_path.read_text()) if timed_path.exists() else {}
                        timed_record.update(status='timeout', timeout_seconds=args.timeout)
                        timed_path.write_text(json.dumps(timed_record, indent=2) + '\n')
                report_path = args.output / f'{grid}_{formulation}.json'
                if report_path.exists():
                    report = json.loads(report_path.read_text())
                else:
                    report = dict(status='failed', error='Child exited without a JSON report')
                summary.append(dict(grid=grid, formulation=formulation, **{key: report[key]
                    for key in ('status', 'stage', 'total_seconds', 'finite_modes', 'error') if key in report}))
                print(f'{grid} {formulation}: {report["status"]}', flush=True)
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return int(failed)
    for suffix in ('modes.csv', 'trajectory.npz'):
        (args.output / f'{args.grid}_{args.formulation}_{suffix}').unlink(missing_ok=True)
    record = dict(status='running', settings=vars(args).copy())
    record['settings']['output'] = str(args.output)
    record['settings']['engine_root'] = str(args.engine_root) if args.engine_root else None
    np_seed = 0
    import numpy as np
    np.random.seed(np_seed)
    record['random_seed'] = np_seed
    from numba.core import event as numba_event
    compilation_timer = numba_event.TimingListener()
    start = time.perf_counter()
    try:
        with numba_event.install_listener('numba:compile', compilation_timer):
            run_case(args, record)
    except KeyboardInterrupt:
        record['status'] = 'interrupted'
        raise
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        record['total_seconds'] = time.perf_counter() - start
        record['compilation_seconds'] = compilation_timer.duration if compilation_timer.done else 0.
        record['calculation_seconds'] = record['total_seconds'] - record['compilation_seconds']
        record['compilation_scope'] = 'Outermost numba:compile events; cache hits excluded'
        record['calculation_scope'] = 'Remaining measured pipeline time: loading, PF, assembly, cache loading, initialization, checks and SSA'
        (args.output / f'{args.grid}_{args.formulation}.json').write_text(json.dumps(record, indent=2) + '\n')
    return int(record['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
