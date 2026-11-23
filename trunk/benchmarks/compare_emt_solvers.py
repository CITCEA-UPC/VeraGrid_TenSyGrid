#!/usr/bin/env python3
"""Run identical IEEE39 multilinear SSA with each EMT capture solver."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from multilinear_ssa import ROOT  # Apply the benchmark's thread/cache defaults.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', choices=['ieee39', 'ieee118'], default='ieee39')
    parser.add_argument('--solvers', nargs='+',
                        choices=['StructuralAD', 'StructuralCompiled', 'Automatic', 'Symbolic'],
                        default=['StructuralAD', 'StructuralCompiled', 'Automatic', 'Symbolic'])
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--steps-per-period', type=int, default=600)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'solver_comparison')
    args = parser.parse_args()
    if args.timeout <= 0 or args.steps_per_period <= 0:
        parser.error('timeout and steps per period must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    previous = json.loads(summary_path.read_text()) if summary_path.exists() else []
    summary = [row for row in previous if row['solver'] not in args.solvers]
    for solver in args.solvers:
        destination = output / solver
        destination.mkdir(exist_ok=True)
        report_path = destination / f'{args.grid}_emt.json'
        report_path.unlink(missing_ok=True)
        command = [sys.executable, '-u', str(Path(__file__).with_name('multilinear_ssa.py')),
                   '--grid', args.grid, '--formulation', 'emt', '--emt-solver', solver,
                   '--steps-per-period', str(args.steps_per_period), '--assessment-periods', '1', '--modes', '6',
                   '--output', str(destination)]
        print(f'Running {solver}', flush=True)
        start = time.perf_counter()
        with (destination / f'{args.grid}_emt.log').open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                returncode = result.returncode
                timed_out = False
            except subprocess.TimeoutExpired:
                returncode = None
                timed_out = True
        record = json.loads(report_path.read_text()) if report_path.exists() else {}
        if timed_out:
            record.update(status='timeout', timeout_seconds=args.timeout)
        elif returncode != 0:
            record['status'] = 'failed'
        record.update(parent_wall_seconds=time.perf_counter() - start, returncode=returncode)
        report_path.write_text(json.dumps(record, indent=2) + '\n')
        row = dict(solver=solver, **{k:record[k] for k in ('status','stage','parent_wall_seconds',
            'build_seconds','ssa_seconds','trajectory_and_jacobian_setup_seconds',
            'floquet_after_capture_seconds','total_seconds','error','max_multiplier_magnitude') if k in record})
        summary.append(row)
        print(f'{solver}: {row.get("status", "unknown")}', flush=True)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    # Compare successful trajectories and modes to the actual StructuralAD run.
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    reference_dir = output / 'StructuralAD'
    baseline = next((row for row in summary if row['solver'] == 'StructuralAD'), {})
    if baseline.get('status') == 'completed':
        reference = np.load(reference_dir / f'{args.grid}_emt_trajectory.npz')['trajectory']
        def modes(directory):
            data = np.genfromtxt(directory / f'{args.grid}_emt_modes.csv', delimiter=',', names=True)
            return data['real_per_s'] + 1j * data['imag_rad_per_s']
        reference_modes = modes(reference_dir)
        for row in summary:
            if row.get('status') != 'completed':
                continue
            directory = output / row['solver']
            trajectory = np.load(directory / f'{args.grid}_emt_trajectory.npz')['trajectory']
            if trajectory.shape == reference.shape:
                scales = np.maximum(1.0, np.max(np.abs(reference), axis=0))
                row['trajectory_scaled_difference_inf'] = float(np.max(np.abs(trajectory-reference)/scales))
            candidate_modes = modes(directory)
            distances = abs(reference_modes[:,None] - candidate_modes[None,:])
            r,c = linear_sum_assignment(distances)
            row['matched_mode_difference_inf_per_s'] = float(np.max(distances[r,c]))
            row['compared_mode_count'] = len(r)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return int(any(row.get('status') != 'completed' for row in summary))


if __name__ == '__main__':
    raise SystemExit(main())
