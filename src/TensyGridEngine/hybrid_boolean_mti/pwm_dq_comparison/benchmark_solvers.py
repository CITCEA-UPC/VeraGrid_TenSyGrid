"""Reproduce PWM solver timings and full switched-converter AC/DC grid comparisons.

Run with PYTHONPATH=src MPLBACKEND=Agg python <this-file>.
Each case runs in an isolated process; first-call and repeat times include fresh
problem construction. Grid setup and simulation are measured separately.
"""
import argparse
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
OUT = HERE / 'benchmark_results'
os.environ.setdefault('VERAGRID_EMT_INIT_CACHE_DIR', '/tmp/tensygrid-pwm-init-cache')


def worker(case):
    if case == 'grid_mti_Symbolic':
        return dict(case=case, status='skipped', reason='MTI uses its dedicated integrator; the EMT Jacobian backend is not a separate MTI solver')
    if case.startswith('pwm_'):
        from TensyGridEngine.hybrid_boolean_mti import compare_pwm_mti_direct as h
        funcs = {
            'pwm_procedural': lambda: h.run_solver_backed_case(True),
            'pwm_live_comparator': lambda: h.run_solver_backed_case(False),
            'pwm_internal_mti': h.run_back_euler_mti_block_case,
            'pwm_scheduled_mti': h.run_back_euler_mti_scheduled_block_case,
        }
        durations = []
        for _ in range(4):
            start = perf_counter()
            result = funcs[case]()
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
        np.savez_compressed(OUT / f'{case}.npz', time=gt, gates=gates)
        return dict(case=case, status='ok', first_call_seconds=durations[0],
                    repeat_seconds=durations[1:], median_repeat_seconds=float(np.median(durations[1:])),
                    points=len(t), label=label)

    from TensyGridEngine.emt.validate_converter_emt_multilinear import simulate_case, build_emt_options
    from VeraGridEngine.Templates.Emt.converter_switched_emt_template import get_switched_emt_converter
    from VeraGridEngine.Templates.Emt.converter_switched_emt_mti_template import get_switched_emt_converter_mti
    from VeraGridEngine.Templates.Emt.converter_switched_emt_multilinear_template import get_switched_emt_converter_multilinear
    from VeraGridEngine.enumerations import EmtSolverTypes, DynamicIntegrationMethod
    _, variant, backend = case.split('_')
    builders = dict(standard=get_switched_emt_converter, mti=get_switched_emt_converter_mti,
                    multilinear=get_switched_emt_converter_multilinear)
    opt = build_emt_options()
    opt.solver_type = EmtSolverTypes[backend]
    from VeraGridEngine.Simulations.EMT.solvers.structural_compiled_solver import StructuralCompiledWarmupPolicy
    opt.compiled_warmup_policy = StructuralCompiledWarmupPolicy.Off
    opt.integration_method = DynamicIntegrationMethod.DaeBackEuler
    opt.init_newton_max_iter = 60
    r = simulate_case(case, builders[variant], opt, use_mti=variant == 'mti')
    signals = {k: v for k, v in asdict(r).items() if isinstance(v, np.ndarray)}
    np.savez_compressed(OUT / f'{case}.npz', **signals)
    valid = r.initialized and r.converged and r.time_arr[-1] >= opt.simulation_time - 1e-10
    valid = valid and all(np.isfinite(v).all() for v in signals.values())
    return dict(case=case, status='ok' if valid else 'invalid', setup_seconds=r.setup_seconds,
                warmup_policy=opt.compiled_warmup_policy.value,
                integrator='EmtMtiBackEulerSolver' if variant == 'mti' else backend,
                solve_seconds=r.solve_seconds, initialized=r.initialized, converged=r.converged,
                final_time=float(r.time_arr[-1]), points=len(r.time_arr))


def compare(results):
    import matplotlib.pyplot as plt
    valid = [r['case'] for r in results if r['status'] == 'ok' and r['case'].startswith('grid_')]
    if not valid:
        return {}
    baseline = 'grid_standard_StructuralCompiled'
    if baseline not in valid:
        baseline = valid[0]
    ref = np.load(OUT / f'{baseline}.npz')
    metrics = {'baseline': baseline, 'cases': {}}
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout='constrained')
    for case in valid:
        trace = np.load(OUT / f'{case}.npz')
        metrics['cases'][case] = {}
        for ax, key in zip(axes.flat, ('i_a', 'i_b', 'i_c', 'i_dc', 'p', 'q')):
            values = np.interp(ref['time_arr'], trace['time_arr'], trace[key])
            diff = values - ref[key]
            metrics['cases'][case][key] = dict(max_abs=float(np.max(np.abs(diff))), rmse=float(np.sqrt(np.mean(diff**2))))
            ax.plot(trace['time_arr'] * 1000, trace[key], label=case.removeprefix('grid_'), linewidth=1)
            ax.set(xlabel='Time [ms]', ylabel=key)
            ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Full switched converter in AC/DC grid: Backward Euler, 10 µs, 2 ms')
    fig.savefig(OUT / 'grid_comparison.png', dpi=150)
    plt.close(fig)
    return metrics


def compare_pwm(results):
    import matplotlib.pyplot as plt
    cases = [r['case'] for r in results if r['status'] == 'ok' and r['case'].startswith('pwm_')]
    cases.sort(key=lambda case: (case != 'pwm_procedural', case))
    grid = np.linspace(0., .004, 40001)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout='constrained')
    metrics = {}
    reference = None
    for case in cases:
        trace = np.load(OUT / f'{case}.npz')
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
    fig.savefig(OUT / 'pwm_comparison.png', dpi=150)
    plt.close(fig)
    (OUT / 'pwm_errors.json').write_text(json.dumps(metrics, indent=2) + '\n')


def write_report(results, metrics):
    lines = ['# PWM and full converter benchmark', '',
             'Run on 25 November 2026 with one BLAS/OpenMP thread. Timings are machine-specific.', '',
             '## PWM solver timing', '',
             'Sinusoidal references, 1 kHz switching, 50 Hz frame, modulation 0.8; '
             '4 ms duration with a nominal 50 µs step. Each call constructs a fresh problem. '
             'Three repeats follow the first call; imports are excluded.', '',
             '| Variant | First call (s) | Median repeat (s) |', '| --- | ---: | ---: |']
    for r in results:
        if r['case'].startswith('pwm_') and r['status'] == 'ok':
            lines.append(f"| {r['case']} | {r['first_call_seconds']:.6f} | {r['median_repeat_seconds']:.6f} |")
    lines += ['', 'These variants have different sampling/event semantics. See `pwm_errors.json` '
              'and `pwm_comparison.png`; timings alone do not establish equivalent solutions.', '',
              'The separate aligned-event test uses identical held references and ramp boundaries '
              'for both sinusoidal and variable-dq inputs. See `aligned_events.log`. '
              'It checks the event equations, rather than the solver adapters.', '',
              '## Full switched converter in an AC/DC grid', '',
              'Three buses: 230 kV AC source behind a Thevenin equivalent and transformer, '
              'a 100 MVA VSC, and a 320 kV DC bus with a 30 MW load. '
              'Power-flow-seeded initialization; full bridge, RL filter, PLL, controls, '
              'DC dynamics and startup handover at 0.1 ms. '
              'Backward Euler, 10 µs nominal step, 2 ms duration (two switching periods; '
              'this is a startup comparison, not a steady-state or stability study).', '',
              'The MTI variant uses `EmtProblemMTI` and `EmtMtiBackEulerSolver`. '
              'Other variants use the named EMT backend. For the short run, '
              'StructuralCompiled uses its supported warmup-off policy (generated Python kernels). '
              'The default adaptive warmup attempt exceeded 300 s during compilation. '
              'Setup includes PF, model assembly, initialization and solver construction. '
              'Simulation includes any compilation deferred until first use. '
              'Persistent caches and concurrent machine load can affect setup times. Grid timings are single runs. '
              'Earlier default-policy attempts are archived in `initial_attempts.json` and `*_adaptive_timeout.log`.', '',
              '| Case | Status | Setup (s) | Simulation (s) |', '| --- | --- | ---: | ---: |']
    for r in results:
        if r['case'].startswith('grid_'):
            setup = f"{r['setup_seconds']:.3f}" if 'setup_seconds' in r else '—'
            solve = f"{r['solve_seconds']:.3f}" if 'solve_seconds' in r else '—'
            lines.append(f"| {r['case']} | {r['status']} | {setup} | {solve} |")
    lines += ['', 'Only runs that initialize, converge, remain finite and reach the requested '
              'end time enter the waveform comparison. Errors and timeouts are retained in '
              '`timings.json` and per-case logs. These status checks do not establish accuracy.', '',
              'Waveforms are interpolated onto the baseline timestamps for max-absolute and RMS '
              'errors; gate discontinuities use previous-value sampling in the PWM comparison. '
              'Grid currents/powers use the model’s per-unit convention.', '']
    if metrics:
        lines += [f"Baseline: `{metrics['baseline']}`.", '',
                  '| Case | Phase-A current max error (pu) | DC current max error (pu) | P max error (pu) |',
                  '| --- | ---: | ---: | ---: |']
        for case, values in metrics['cases'].items():
            lines.append(f"| {case} | {values['i_a']['max_abs']:.6g} | {values['i_dc']['max_abs']:.6g} | {values['p']['max_abs']:.6g} |")
    lines += ['', 'The full-grid MTI run failed initialization and is excluded from agreement comparisons. '
              'Automatic and StructuralAD crashed during native compilation. '
              'The multilinear converter differs substantially from the standard converter; '
              'its lower runtime does not establish an equivalent solution.', '', '## Reproduce', '', 'From the repository root:', '', '```sh',
              'PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/pwm-mpl \\',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\',
              'python src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/benchmark_solvers.py --timeout 300',
              '```', '', 'Use `--cases CASE...` to retry selected cases while retaining other saved results. '
              'Local changes were present in the working tree during these runs; this is not a clean-checkout benchmark.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--grid-only', action='store_true')
    parser.add_argument('--cases', nargs='+', help='Run selected cases and retain other saved results')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.worker:
        import faulthandler
        faulthandler.dump_traceback_later(60, repeat=True)
        try:
            result = worker(args.worker)
        except Exception as exc:
            traceback.print_exc()
            result = dict(case=args.worker, status='error', error=str(exc))
        (OUT / f'{args.worker}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    cases = ['pwm_procedural', 'pwm_live_comparator', 'pwm_internal_mti', 'pwm_scheduled_mti']
    cases += [f'grid_{variant}_{backend}' for variant in ('standard', 'mti', 'multilinear')
              for backend in (('StructuralCompiled',) if variant == 'mti' else ('StructuralCompiled', 'Symbolic'))]
    cases += ['grid_standard_Automatic', 'grid_standard_StructuralAD']
    if args.grid_only:
        cases = [c for c in cases if c.startswith('grid_')]
    if args.cases:
        cases = args.cases
    results = []
    if args.cases and (OUT / 'timings.json').exists():
        results = [r for r in json.loads((OUT / 'timings.json').read_text())['results'] if r['case'] not in cases]
    elif args.grid_only and (OUT / 'timings.json').exists():
        results = [r for r in json.loads((OUT / 'timings.json').read_text())['results'] if r['case'].startswith('pwm_')]
    for case in cases:
        print(f'Running {case}', flush=True)
        result_path = OUT / f'{case}.json'
        result_path.unlink(missing_ok=True)
        with (OUT / f'{case}.log').open('w') as log:
            try:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', case, '--timeout', str(args.timeout)],
                               stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
                result = json.loads(result_path.read_text())
            except Exception as exc:
                result = dict(case=case, status='error', error=str(exc))
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
        print(result, flush=True)
        (OUT / 'timings.json').write_text(json.dumps(dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
            revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            working_tree='benchmark run includes local uncommitted changes',
            threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
            settings=dict(pwm_duration_s=.004, pwm_step_s=50e-6, grid_duration_s=.002, grid_step_s=10e-6,
                          pwm_timing='one first call + three repeats, includes setup', grid_timing='one fresh process, setup separate; simulation includes first-use compilation'),
            results=results), indent=2) + '\n')
    compare_pwm(results)
    metrics = compare(results)
    (OUT / 'grid_errors.json').write_text(json.dumps(metrics, indent=2) + '\n')
    write_report(results, metrics)


if __name__ == '__main__':
    main()
