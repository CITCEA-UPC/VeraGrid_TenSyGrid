#!/usr/bin/env python3
"""Measure compilation and remaining run time for the four verified RMS/EMT cases."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from multilinear_ssa import ROOT, source_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'trunk/benchmarks/results/four_cases')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name('multilinear_ssa.py')
    engine_hash = source_fingerprint(ROOT/'src/VeraGridEngine')
    runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
    report = dict(status='running', cache_policy='Fresh separate Numba disk-cache directory per case; other caches unchanged',
                  engine_source_sha256=engine_hash, benchmark_source_sha256=runner_hash, cases={})
    try:
        for grid in ('ieee39', 'ieee118'):
            for formulation in ('rms', 'emt'):
                folder = args.output/f'{grid}_{formulation}'
                folder.mkdir(exist_ok=True)
                environment = os.environ.copy()
                for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
                    environment[name] = '1'
                cache = tempfile.mkdtemp(prefix=f'veragrid-{grid}-{formulation}-numba-')
                environment['NUMBA_CACHE_DIR'] = cache
                command = [sys.executable, '-u', str(runner), '--grid', grid, '--formulation', formulation,
                    '--rms-spectrum', 'reduced', '--emt-problem', 'Multilinear', '--modes', '6',
                    '--steps-per-period', '600', '--assessment-periods', '1', '--krylov-dim', '24',
                    '--output', str(folder)]
                print(f'Running {grid}/{formulation} with fresh Numba cache', flush=True)
                with (folder/'run.log').open('w') as stream:
                    child = subprocess.run(command, env=environment, stdout=stream,
                        stderr=subprocess.STDOUT, timeout=600, cwd=ROOT)
                record = json.loads((folder/f'{grid}_{formulation}.json').read_text())
                if child.returncode or record.get('status') != 'completed':
                    raise RuntimeError(f'{grid}/{formulation}: {record.get("error", "failed")}')
                if record['engine_source_sha256'] != engine_hash or record['benchmark_source_sha256'] != runner_hash:
                    raise RuntimeError('Source changed during phase measurement')
                if not Path(record['engine_module']).resolve().is_relative_to(ROOT/'src'):
                    raise RuntimeError('Engine imported from another checkout')
                checks = ('equilibrium_validated','representation_validated','eigenpairs_validated') if formulation=='rms' else ('initialization_validated','periodicity_validated','three_phase_pf_converged')
                if not record['balanced_pf_converged'] or not all(record.get(key) for key in checks):
                    raise RuntimeError('A required operating-point or numerical check failed')
                measured = [record[key] for key in ('compilation_seconds','calculation_seconds','total_seconds')]
                if not all(math.isfinite(value) and value >= 0 for value in measured):
                    raise RuntimeError('Invalid phase times')
                if abs(measured[0]+measured[1]-measured[2]) > 1e-9:
                    raise RuntimeError('Phase times do not sum to total')
                report['cases'][f'{grid}/{formulation}'] = {key:record[key] for key in (
                    'compilation_seconds','calculation_seconds','total_seconds','ssa_seconds','finite_modes')}
                report['cases'][f'{grid}/{formulation}']['numba_cache_directory'] = cache
                print(f'{grid}/{formulation}: JIT {measured[0]:.2f} s, other {measured[1]:.2f} s, total {measured[2]:.2f} s', flush=True)
                (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
