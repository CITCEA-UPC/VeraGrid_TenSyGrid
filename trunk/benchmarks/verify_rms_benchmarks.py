#!/usr/bin/env python3
"""Verify all finite RMS modes on IEEE39 and IEEE118 from unchanged descriptor pencils."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multilinear_ssa import ROOT, source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'trunk/benchmarks/results/rms')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name('multilinear_ssa.py')
    helper = Path(__file__).with_name('rms_descriptor.py')
    report = dict(status='running', engine_source_sha256=source_fingerprint(ROOT/'src/VeraGridEngine'),
        benchmark_source_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        rms_descriptor_source_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(), cases={})
    environment = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[key] = '1'
    started = time.perf_counter()
    try:
        for grid in ('ieee39', 'ieee118'):
            folder = args.output/grid
            folder.mkdir(exist_ok=True)
            command = [sys.executable, '-u', str(runner), '--grid', grid, '--formulation', 'rms',
                       '--rms-spectrum', 'reduced', '--rms-init', 'explicit', '--output', str(folder)]
            with (folder/'run.log').open('w') as stream:
                child = subprocess.run(command, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, timeout=300, cwd=ROOT)
            result = json.loads((folder/f'{grid}_rms.json').read_text())
            if child.returncode or result['status'] != 'completed':
                raise RuntimeError(f'{grid}: {result.get("error", "run failed")}')
            for key in ('engine_source_sha256', 'benchmark_source_sha256', 'rms_descriptor_source_sha256'):
                if result.get(key) != report[key]:
                    raise RuntimeError(f'{grid}: source fingerprint mismatch')
            if not Path(result['engine_module']).resolve().is_relative_to(ROOT/'src'):
                raise RuntimeError(f'{grid}: engine imported from a different checkout')
            if not all(result.get(key) for key in ('balanced_pf_converged', 'equilibrium_validated',
                                                 'representation_validated', 'eigenpairs_validated')):
                raise RuntimeError(f'{grid}: a validation stage did not pass')
            if (result['descriptor_regularization'] != 0.
                    or result['finite_modes'] != result['reduced_dimension']
                    or result['power_flow_voltage_difference_inf'] > 1e-10):
                raise RuntimeError(f'{grid}: altered pencil, incomplete spectrum or changed PF voltage')
            report['cases'][grid] = {key: result[key] for key in (
                'total_seconds', 'build_seconds', 'ssa_seconds', 'matrix_build_seconds',
                'finite_modes', 'descriptor_dimension', 'reduced_dimension',
                'initial_residual_inf', 'symbolic_initial_residual_inf', 'representation_error_inf',
                'power_flow_voltage_difference_inf', 'eigenpair_backward_error_max', 'max_real_per_s')}
            print(f'{grid}: {result["total_seconds"]:.2f} s total, '
                  f'{result["ssa_seconds"]:.3f} s SSA, {result["finite_modes"]} finite modes', flush=True)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['verification_seconds'] = time.perf_counter()-started
        (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
