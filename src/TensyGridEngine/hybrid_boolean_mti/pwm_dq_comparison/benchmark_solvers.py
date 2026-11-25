"""Reproduce PWM solver timings and full switched-converter AC/DC grid comparisons.

Run with PYTHONPATH=src MPLBACKEND=Agg python <this-file>.
Each case runs in an isolated process; first-call and repeat times include fresh
problem construction. Grid setup and simulation are measured separately.
"""
import argparse
import hashlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault('VERAGRID_EMT_INIT_CACHE_DIR', '/tmp/tensygrid-pwm-init-cache')


def worker(case, output: Path, pwm_step: float = 50e-6,
           grid_step: float = 10e-6, grid_duration: float = .002, soft_lambda: float = 1e-6):
    if case == 'grid_mti_Symbolic':
        return dict(case=case, status='skipped', reason='MTI uses its dedicated integrator; the EMT Jacobian backend is not a separate MTI solver')
    if case.startswith('pwm_'):
        from TensyGridEngine.hybrid_boolean_mti import compare_pwm_mti_direct as h
        funcs = {
            'pwm_procedural': lambda timing: h.run_solver_backed_case(True, h=pwm_step, timings=timing),
            'pwm_live_comparator': lambda timing: h.run_solver_backed_case(False, h=pwm_step, timings=timing),
            'pwm_internal_mti': lambda timing: h.run_back_euler_mti_block_case(h=pwm_step, timings=timing),
            'pwm_scheduled_mti': lambda timing: h.run_back_euler_mti_scheduled_block_case(h=pwm_step, timings=timing),
        }
        durations = []
        phases = []
        for _ in range(4):
            start = perf_counter()
            timing = {}
            result = funcs[case](timing)
            phases.append(timing)
            durations.append(perf_counter() - start)
            if case in ('pwm_procedural', 'pwm_live_comparator'):
                t, _, gt, gates, initialized, converged, label = result
            elif case == 'pwm_internal_mti':
                t, _, gates, initialized, converged, label = result
                gt = t
            else:
                t, gates, _, initialized, converged, label = result
                gt = t
            if not initialized or not converged or not np.isfinite(gates).all() or t[-1] < .004 - 1e-10:
                raise RuntimeError('PWM run failed initialization, convergence, finiteness or end-time checks')
        np.savez_compressed(output / f'{case}.npz', time=gt, gates=gates)
        return dict(case=case, status='ok', first_call_seconds=durations[0],
                    repeat_seconds=durations[1:], median_repeat_seconds=float(np.median(durations[1:])),
                    points=len(t), label=label, nominal_step_seconds=pwm_step,
                    first_setup_seconds=phases[0]["setup_seconds"], first_solve_seconds=phases[0]["solve_seconds"],
                    median_setup_seconds=float(np.median([p["setup_seconds"] for p in phases[1:]])),
                    median_solve_seconds=float(np.median([p["solve_seconds"] for p in phases[1:]])))

    from TensyGridEngine.emt.validate_converter_emt_multilinear import simulate_case, build_emt_options
    from VeraGridEngine.Templates.Emt.bridge_2level_3ph_emt_soft_sign_template import get_switched_emt_converter_soft_sign
    from VeraGridEngine.Templates.Emt.converter_switched_emt_template import get_switched_emt_converter
    from VeraGridEngine.Templates.Emt.converter_switched_emt_mti_template import get_switched_emt_converter_mti
    from VeraGridEngine.Templates.Emt.converter_switched_emt_multilinear_template import get_switched_emt_converter_multilinear
    from VeraGridEngine.enumerations import EmtSolverTypes, DynamicIntegrationMethod
    _, variant, backend = case.split('_')
    builders = dict(standard=get_switched_emt_converter, mti=get_switched_emt_converter_mti,
                    multilinear=get_switched_emt_converter_multilinear, softsign=get_switched_emt_converter_soft_sign)
    opt = build_emt_options()
    opt.solver_type = EmtSolverTypes[backend]
    from VeraGridEngine.Simulations.EMT.solvers.structural_compiled_solver import StructuralCompiledWarmupPolicy
    opt.compiled_warmup_policy = StructuralCompiledWarmupPolicy.Off
    opt.integration_method = DynamicIntegrationMethod.DaeBackEuler
    opt.init_newton_max_iter = 60
    opt.time_step = grid_step
    opt.simulation_time = grid_duration
    r = simulate_case(case, builders[variant], opt, use_mti=variant == 'mti',
                      model_parameters={'soft_sign_lambda': soft_lambda} if variant == 'softsign' else None)
    signals = {k: v for k, v in asdict(r).items() if isinstance(v, np.ndarray)}
    np.savez_compressed(output / f'{case}.npz', **signals)
    valid = r.initialized and r.converged and r.time_arr[-1] >= opt.simulation_time - 1e-10
    valid = valid and all(np.isfinite(v).all() for v in signals.values())
    valid = valid and r.initialization_residual < 1e-7
    return dict(case=case, status='ok' if valid else 'invalid', setup_seconds=r.setup_seconds,
                warmup_policy=opt.compiled_warmup_policy.value,
                integrator='EmtMtiBackEulerSolver' if variant == 'mti' else backend,
                initialization_residual=r.initialization_residual,
                nominal_step_seconds=grid_step, requested_duration_seconds=grid_duration,
                soft_sign_lambda=soft_lambda if variant == "softsign" else None,
                event_bisect_iter=12 if variant == 'mti' else None,
                solve_seconds=r.solve_seconds, initialized=r.initialized, converged=r.converged,
                final_time=float(r.time_arr[-1]), points=len(r.time_arr))


def compare(results, output: Path):
    import matplotlib.pyplot as plt
    valid = [r['case'] for r in results if r['status'] == 'ok' and r['case'].startswith('grid_')]
    order = {'standard': 0, 'multilinear': 1, 'mti': 2, 'softsign': 3}
    valid.sort(key=lambda case: order[case.split('_')[1]])
    if not valid:
        return {}
    baseline = 'grid_standard_StructuralCompiled'
    if baseline not in valid:
        (output / 'grid_comparison.png').unlink(missing_ok=True)
        return {}
    ref = np.load(output / f'{baseline}.npz')
    metrics = {'baseline': baseline, 'cases': {}}
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout='constrained')
    for case in valid:
        trace = np.load(output / f'{case}.npz')
        metrics['cases'][case] = {}
        for ax, key in zip(axes.flat, ('i_a', 'v_dc', 'i_dc', 'p', 'p_filtered', 'q_filtered')):
            values = np.interp(ref['time_arr'], trace['time_arr'], trace[key])
            diff = values - ref[key]
            metrics['cases'][case][key] = dict(max_abs=float(np.max(np.abs(diff))), rmse=float(np.sqrt(np.mean(diff**2))))
            ax.plot(trace['time_arr'] * 1000, trace[key], label={'standard': 'Procedural PWM', 'multilinear': 'Trig-state model (procedural PWM)',
                   'mti': 'MTI switching', 'softsign': 'Soft-sign'}[case.split('_')[1]], linewidth=1)
            labels = dict(i_a='Phase A current [pu]', v_dc='DC bus voltage [pu]',
                          i_dc='DC current [pu]', p='Instantaneous active power [pu]',
                          p_filtered='Filtered active power [pu]', q_filtered='Filtered reactive power [pu]')
            ax.set(xlabel='Time [ms]', ylabel=labels[key])
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat:
        ax.axvline(.1, color='gray', linestyle=':', linewidth=.8)
    fig.suptitle('Matched converter models: 5 kHz PWM, Backward Euler', fontsize=12)
    fig.savefig(output / 'grid_comparison.png', dpi=150)
    plt.close(fig)
    return metrics


def compare_pwm(results, output: Path):
    import matplotlib.pyplot as plt
    cases = [r['case'] for r in results if r['status'] == 'ok' and r['case'].startswith('pwm_')]
    if not cases:
        return
    cases.sort(key=lambda case: (case != 'pwm_procedural', case))
    grid = np.linspace(0., .004, 40001)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout='constrained')
    metrics = {}
    reference = None
    for case in cases:
        trace = np.load(output / f'{case}.npz')
        index = np.clip(np.searchsorted(trace['time'], grid, side='right') - 1, 0, len(trace['time']) - 1)
        gates = trace['gates'][index, :3]
        if reference is None:
            reference = gates
        metrics[case] = dict(mismatch_fraction=np.mean(np.abs(gates-reference) > .5, axis=0).tolist(),
                            transitions=np.count_nonzero(np.diff(trace['gates'][:, :3], axis=0), axis=0).tolist())
        for phase, ax in enumerate(axes):
            ax.step(grid * 1000, gates[:, phase], where='post', label=case.removeprefix('pwm_'), alpha=.75)
            ax.set_ylabel('Gate ' + 'ABC'[phase])
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=8, ncol=4)
    axes[-1].set_xlabel('Time [ms]')
    fig.suptitle('Solver-backed PWM comparison; variants use different event/input semantics')
    fig.savefig(output / 'pwm_comparison.png', dpi=150)
    plt.close(fig)
    (output / 'pwm_errors.json').write_text(json.dumps(metrics, indent=2) + '\n')


def plot_timings(results, output: Path):
    """Compare setup and integration costs, retaining status labels on grid runs."""
    import matplotlib.pyplot as plt
    pwm = [r for r in results if r['case'].startswith('pwm_') and r['status'] == 'ok']
    grid = [r for r in results if r['case'].startswith('grid_') and 'setup_seconds' in r]
    both = bool(pwm and grid)
    fig, axes = plt.subplots(1, 2 if both else 1, figsize=(12 if both else 7, 4),
                             squeeze=False, layout='constrained')
    axis = axes[0, 0]
    names = [r['case'].removeprefix('pwm_') for r in pwm]
    setup = np.array([r['median_setup_seconds'] * 1000 for r in pwm])
    solve = np.array([r['median_solve_seconds'] * 1000 for r in pwm])
    axis.barh(names, setup, label='Setup')
    axis.barh(names, solve, left=setup, label='Integration')
    axis.set(xlabel='Median elapsed time [ms]', title='Standalone PWM: fresh-model repeats')
    axis.legend()
    axis.grid(axis='x', alpha=.2)
    if grid:
        axis = axes[0, 1 if both else 0]
        if not pwm:
            axis.clear()
        names = [{'standard': 'Procedural PWM', 'multilinear': 'Trig-state model', 'mti': 'MTI switching', 'softsign': 'Soft-sign'}[r['case'].split('_')[1]] +
                 ('' if r['status'] == 'ok' else ' (invalid)') for r in grid]
        setup = np.array([r['setup_seconds'] for r in grid])
        solve = np.array([r['solve_seconds'] for r in grid])
        axis.barh(names, setup, label='PF + model + solver setup')
        axis.barh(names, solve, left=setup, label='Integration')
        axis.set(xlabel='Elapsed time [s]', title='Full converter: single runs')
        axis.legend(fontsize=8)
        axis.grid(axis='x', alpha=.2)
    fig.savefig(output / 'timing_comparison.png', dpi=160)
    plt.close(fig)


def write_report(results, metrics, output: Path):
    """Write current measurements and limitations without reusing historical conclusions."""
    pwm = [r for r in results if r['case'].startswith('pwm_') and r['status'] == 'ok']
    grid = [r for r in results if r['case'].startswith('grid_')]
    lines = ['# Matched PWM converter benchmarks', '',
             'Upstream source: SanPen/VeraGrid `3fc0e970c`. Local ports retain the existing '
             'module layout. Raw trajectories, status checks and per-case logs accompany this report.', '',
             '## Standalone PWM', '',
             '4 ms, 1 kHz carrier, 50 Hz sinusoidal references, modulation 0.8. '
             'One first call and three repeated calls construct fresh models. Imports are excluded. '
             'Setup and integration are measured separately. Times depend on machine load and caches.', '',
             '| Variant | Step (µs) | First total (ms) | Median setup (ms) | Median integration (ms) | Median total (ms) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in pwm:
        lines.append(f"| {r['case']} | {r['nominal_step_seconds']*1e6:g} | {r['first_call_seconds']*1e3:.3f} | "
                     f"{r['median_setup_seconds']*1e3:.3f} | {r['median_solve_seconds']*1e3:.3f} | "
                     f"{r['median_repeat_seconds']*1e3:.3f} |")
    lines += ['', 'The live comparator uses continuous references. Procedural PWM and both '
              'standalone MTI adapters request the same held sinusoidal reference samples. '
              'Runtime event handling and output recording can still differ. The 0.1 µs plotting '
              'grid is a previous-value reconstruction, not the integration step. '
              'The event-equation aligned test is a separate check and does not time the MTI integrator.', '']
    error_path = output / 'pwm_errors.json'
    if error_path.exists():
        errors = json.loads(error_path.read_text())
        lines += ['| Variant | Gate mismatch A / B / C (%) |', '| --- | --- |']
        for case, values in errors.items():
            mismatch = ' / '.join(f'{value*100:.3f}' for value in values['mismatch_fraction'])
            lines.append(f'| {case} | {mismatch} |')
    if grid:
        lines += ['', '## Full converter in the AC/DC grid', '',
                  'Three buses, 100 MVA base, 230 kV AC and 320 kV DC, 30 MW DC load. '
                  'The upstream converter defaults use a 5 kHz carrier starting at an extremum, '
                  'longer measurement/PLL filtering and startup gain scheduling. '
                  f"Backward Euler, {grid[0].get('nominal_step_seconds', 1e-5)*1e6:g} µs nominal step, "
                  f"{grid[0].get('requested_duration_seconds', .002)*1000:g} ms horizon, handover at 0.1 ms. "
                  'All variants use the same corrected power base, loss convention and nominal limits. '
                  'The MTI bridge retains a continuous comparator and integrated carrier; '
                  'it does not use the standalone scheduled-MTI adapter. The multilinear row is a '
                  'trig-state/auxiliary-equation converter model using the same procedural PWM. '
                  'Its trig states now follow the PLL angular rate and its PWM phase advance is an '
                  'algebraic rotation. MTI uses the same carrier-synchronized held modulation inputs. '
                  'The soft-sign variant changes only the comparator and keeps the shared controls. '
                  'Soft smoothing and finite-step integration remain approximation differences.', '',
                  'Soft-sign uses `gate = (1 + margin / sqrt(margin**2 + lambda)) / 2`, '
                  'where `margin = held_modulation - carrier`. The default `lambda=1e-6` '
                  'has normalized width 0.001 (about 50 ns at 5 kHz). The nominal 10 µs '
                  'step does not resolve that transition; convergence must be checked before '
                  'claiming equivalence to hard switching.', '',
                  'StructuralCompiled uses warmup off; MTI uses its dedicated integrator. '
                  'Grid times are single runs. MTI uses 12 event bisections. Initialization residual must be below 1e-7, '
                  'initialization/convergence flags must pass, values must be finite and the '
                  'requested final time must be reached to enter the comparison.', '',
                  '| Case | Status | Initial algebraic residual | Setup (s) | Integration (s) |',
                  '| --- | --- | ---: | ---: | ---: |']
        if not any(r['case'] == 'grid_standard_StructuralCompiled' and r['status'] == 'ok' for r in grid):
            lines += ['Procedural reference did not complete successfully. No full-horizon error comparison is reported.', '']
        for r in grid:
            setup = f"{r['setup_seconds']:.3f}" if 'setup_seconds' in r else '—'
            solve = f"{r['solve_seconds']:.3f}" if 'solve_seconds' in r else '—'
            residual = f"{r['initialization_residual']:.3g}" if 'initialization_residual' in r else '—'
            lines.append(f"| {r['case']} | {r['status']} | {residual} | {setup} | {solve} |")
    if metrics:
        lines += ['', f"Baseline: `{metrics['baseline']}`. Continuous outputs are interpolated onto its timestamps.", '',
                  '| Case | Max phase-A current error (pu) | Max DC voltage error (pu) | Max filtered P error (pu) |',
                  '| --- | ---: | ---: | ---: |']
        for case, values in metrics['cases'].items():
            lines.append(f"| {case} | {values['i_a']['max_abs']:.6g} | {values['v_dc']['max_abs']:.6g} | "
                         f"{values['p_filtered']['max_abs']:.6g} |")
    lines += ['', '## Reproduce', '', 'From the repository root, in the VeraGrid Python environment:', '', '```sh',
              'PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/pwm-mpl ' + chr(92),
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ' + chr(92),
              'python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_solvers.py --timeout 400 '
              + (f"--grid-step-us {grid[0]['nominal_step_seconds']*1e6:g} "
                 f"--grid-duration-ms {grid[0]['requested_duration_seconds']*1000:g} " if grid else '')
              + f'--output {output}',
              '```', '', 'Use `--output PATH` for separate results, `--pwm-step-us VALUE` to refine the standalone step, '
              'and `--cases CASE...` for selected cases. The original results remain in the parent directory. '
              'Initialization consistency is not a claim of PWM-periodic steady state, nor does convergence alone '
              'establish equivalent solutions.', '']
    if (output / 'VALIDATION.md').exists():
        lines += ['See [validation and failed-attempt notes](VALIDATION.md) and the '
                  '[5 us refinement](step_5us/README.md). The procedural refinement fails at '
                  '1.215 ms, so full-horizon step convergence is not established.', '']
    (output / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--output', type=Path, default=HERE / 'benchmark_results' / 'matched_pwm_models')
    parser.add_argument('--pwm-step-us', type=float, default=50.)
    parser.add_argument('--grid-step-us', type=float, default=10.)
    parser.add_argument('--grid-duration-ms', type=float, default=2.)
    parser.add_argument('--soft-sign-lambda', type=float, default=1e-6)
    parser.add_argument('--grid-only', action='store_true')
    parser.add_argument('--cases', nargs='+', help='Run selected cases and retain other saved results')
    parser.add_argument('--timeout', type=float, default=400)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        import faulthandler
        faulthandler.dump_traceback_later(max(60., args.timeout - 5.), repeat=True)
        try:
            result = worker(args.worker, output, args.pwm_step_us / 1e6, args.grid_step_us / 1e6,
                            args.grid_duration_ms / 1e3, args.soft_sign_lambda)
        except Exception as exc:
            traceback.print_exc()
            result = dict(case=args.worker, status='error', error=str(exc))
        (output / f'{args.worker}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    cases = ['pwm_procedural', 'pwm_live_comparator', 'pwm_internal_mti', 'pwm_scheduled_mti']
    cases += [f'grid_{variant}_{backend}' for variant in ('standard', 'multilinear', 'mti', 'softsign')
              for backend in (('StructuralCompiled',) if variant == 'mti' else ('StructuralCompiled',))]

    if args.grid_only:
        cases = [c for c in cases if c.startswith('grid_')]
    if args.cases:
        cases = args.cases
    results = []
    if args.cases and (output / 'timings.json').exists():
        results = [r for r in json.loads((output / 'timings.json').read_text())['results'] if r['case'] not in cases]
    elif args.grid_only and (output / 'timings.json').exists():
        results = [r for r in json.loads((output / 'timings.json').read_text())['results'] if r['case'].startswith('pwm_')]
    for case in cases:
        print(f'Running {case}', flush=True)
        result_path = output / f'{case}.json'
        result_path.unlink(missing_ok=True)
        with (output / f'{case}.log').open('w') as log:
            try:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', case, '--timeout', str(args.timeout), '--output', str(output), '--pwm-step-us', str(args.pwm_step_us),
                                '--grid-step-us', str(args.grid_step_us), '--grid-duration-ms', str(args.grid_duration_ms),
                                '--soft-sign-lambda', str(args.soft_sign_lambda)],
                               stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
                result = json.loads(result_path.read_text())
            except Exception as exc:
                result = dict(case=case, status='error', error=str(exc))
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
        print(result, flush=True)
        # Save usable plots as soon as their runs are available.
        if case == 'pwm_scheduled_mti':
            compare_pwm(results, output)
        if case.startswith('grid_'):
            interim_metrics = compare(results, output)
            (output / 'grid_errors.json').write_text(json.dumps(interim_metrics, indent=2) + '\n')
            write_report(results, interim_metrics, output)
        (output / 'timings.json').write_text(json.dumps(dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
            revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            working_tree='benchmark run includes local uncommitted changes',
            engine_patch_sha256=hashlib.sha256(subprocess.check_output([
                'git', 'diff', 'HEAD', '--', 'src/VeraGridEngine/Templates/Emt',
                'src/VeraGridEngine/Simulations/EMT/problems',
                'src/VeraGridEngine/Simulations/Rms/numerical/back_euler_mti.py'])).hexdigest(),
            threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
            settings=dict(grid_pwm_hz=5000., pwm_duration_s=.004, pwm_step_s=args.pwm_step_us / 1e6, grid_duration_s=args.grid_duration_ms / 1e3, grid_step_s=args.grid_step_us / 1e6,
                          pwm_timing='one first call + three repeats, includes setup', grid_timing='one fresh process, setup separate; simulation includes first-use compilation'),
            results=results), indent=2) + '\n')
    compare_pwm(results, output)
    metrics = compare(results, output)
    (output / 'grid_errors.json').write_text(json.dumps(metrics, indent=2) + '\n')
    write_report(results, metrics, output)
    plot_timings(results, output)


if __name__ == '__main__':
    main()
