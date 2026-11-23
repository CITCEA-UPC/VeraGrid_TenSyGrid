#!/usr/bin/env python3
"""Run and cross-check both EMT backends on IEEE39 and IEEE118, sequentially."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from multilinear_ssa import ROOT, GRIDS, source_fingerprint


def check_case(record, grid, backend, engine_digest, runner_digest):
    """Reject incomplete, unvalidated or incorrectly sourced measurements."""
    expected_class = 'EmtProblemDae' if backend == 'CurrentBalance' else 'EmtProblemMultilinear'
    if record.get('status') != 'completed':
        raise RuntimeError(f'{grid}/{backend}: {record.get("error", record.get("status"))}')
    if record.get('problem_class') != expected_class:
        raise RuntimeError(f'{grid}/{backend}: wrong problem implementation')
    if not Path(record['engine_module']).resolve().is_relative_to(ROOT / 'src'):
        raise RuntimeError(f'{grid}/{backend}: engine imported from a different checkout')
    if (record.get('engine_source_sha256') != engine_digest
            or record.get('benchmark_source_sha256') != runner_digest):
        raise RuntimeError(f'{grid}/{backend}: source hashes changed')
    expected_grid_digest = hashlib.sha256(GRIDS[grid].read_bytes()).hexdigest()
    if record.get('grid_sha256') != expected_grid_digest:
        raise RuntimeError(f'{grid}/{backend}: wrong or changed grid file')
    if not record.get('initialization_validated'):
        raise RuntimeError(f'{grid}/{backend}: initialization was not validated')
    if not record.get('periodicity_validated') or record.get('finite_modes') != 6:
        raise RuntimeError(f'{grid}/{backend}: missing validated capture or six finite modes')
    residual = record['initialization_report']['final_residual_inf']
    if not np.isfinite(residual) or residual > 1e-6:
        raise RuntimeError(f'{grid}/{backend}: initial DAE residual {residual} exceeds 1e-6')
    if not record.get('balanced_pf_converged') or not record.get('three_phase_pf_converged'):
        raise RuntimeError(f'{grid}/{backend}: a power flow did not converge')


def compare_cases(first, second, grid):
    """Compare trajectories and unordered mode families, including conjugate cutoffs."""
    a = np.genfromtxt(first / f'{grid}_emt_modes.csv', delimiter=',', names=True)
    b = np.genfromtxt(second / f'{grid}_emt_modes.csv', delimiter=',', names=True)
    x = a['real_per_s'] + 1j*a['imag_rad_per_s']
    y = b['real_per_s'] + 1j*b['imag_rad_per_s']
    distances = np.minimum(abs(x[:, None] - y), abs(x[:, None] - y.conjugate()))
    rows, columns = linear_sum_assignment(distances)
    mode_error = float(np.max(distances[rows, columns]))
    with np.load(first / f'{grid}_emt_trajectory.npz') as left:
        with np.load(second / f'{grid}_emt_trajectory.npz') as right:
            if left['trajectory'].shape != right['trajectory'].shape:
                raise RuntimeError(f'{grid}: different equation dimensions')
            trajectory_error = float(np.max(abs(left['trajectory'] - right['trajectory'])))
            if not np.array_equal(left['times'], right['times']):
                raise RuntimeError(f'{grid}: captured time grids differ')
    if mode_error > 1e-6 or trajectory_error > 1e-8:
        raise RuntimeError(f'{grid}: backend mismatch, modes={mode_error}, trajectory={trajectory_error}')
    return dict(max_mode_family_difference_per_s=mode_error,
                max_trajectory_difference=trajectory_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'trunk/benchmarks/results/end_to_end')
    parser.add_argument('--timeout', type=float, default=600, help='Maximum seconds for each full analysis')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name('multilinear_ssa.py')
    engine_digest = source_fingerprint(ROOT / 'src/VeraGridEngine')
    runner_digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    report = dict(status='running', engine_source_sha256=engine_digest,
                  benchmark_source_sha256=runner_digest,
                  verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checkout=str(ROOT), cases=dict(), comparisons=dict())
    environment = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[name] = '1'
    started = time.perf_counter()
    try:
        for grid in ('ieee39', 'ieee118'):
            folders = []
            for backend in ('Multilinear', 'CurrentBalance'):
                folder = args.output / f'{grid}_{backend}'
                folder.mkdir(exist_ok=True)
                folders.append(folder)
                command = [sys.executable, '-u', str(runner), '--grid', grid, '--formulation', 'emt',
                           '--emt-problem', backend, '--emt-models', 'auxiliary',
                           '--steps-per-period', '600', '--assessment-periods', '1',
                           '--modes', '6', '--krylov-dim', '24', '--output', str(folder)]
                print(f'Running {grid}/{backend}', flush=True)
                with (folder / 'run.log').open('w') as stream:
                    process = subprocess.run(command, env=environment, stdout=stream,
                        stderr=subprocess.STDOUT, timeout=args.timeout)
                record = json.loads((folder / f'{grid}_emt.json').read_text())
                check_case(record, grid, backend, engine_digest, runner_digest)
                if process.returncode:
                    raise RuntimeError(f'{grid}/{backend}: child exited {process.returncode}')
                report['cases'][f'{grid}/{backend}'] = {key: record[key] for key in (
                    'total_seconds', 'ssa_seconds', 'build_seconds', 'last_period_scaled_closure_inf',
                    'max_real_per_s', 'max_multiplier_magnitude', 'states', 'algebraics')}
                print(f'Completed {grid}/{backend}: {record["total_seconds"]:.2f} s', flush=True)
                (args.output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            report['comparisons'][grid] = compare_cases(folders[0], folders[1], grid)
        if source_fingerprint(ROOT / 'src/VeraGridEngine') != engine_digest:
            raise RuntimeError('Engine source changed during verification')
        report['status'] = 'passed'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        raise
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['verification_seconds'] = time.perf_counter() - started
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
