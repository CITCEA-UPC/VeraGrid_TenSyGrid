"""Summarize error growth and terminal behavior in a saved converter benchmark."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze(output: Path) -> dict:
    results = json.loads((output / 'timings.json').read_text())['results']
    baseline = 'grid_standard_StructuralCompiled'
    valid = {r['case']: r for r in results if r['status'] == 'ok' and r['case'].startswith('grid_')}
    if baseline not in valid:
        raise ValueError('A completed procedural reference is required')
    ref = np.load(output / f'{baseline}.npz')
    time = ref['time_arr']
    names = {'standard': 'Procedural PWM', 'multilinear': 'Trig-state model',
             'mti': 'MTI switching', 'softsign': 'Soft-sign'}
    colors = {'standard': 'C0', 'multilinear': 'C1', 'mti': 'C2', 'softsign': 'C3'}
    keys = ('i_a', 'v_dc', 'p_filtered')
    labels = ('Phase-A current difference [pu]', 'DC-voltage difference [pu]',
              'Filtered-P difference [pu]')
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    summary = {'baseline': baseline, 'cases': {}}
    for variant in names:
        case = f'grid_{variant}_StructuralCompiled'
        if case not in valid:
            continue
        trace = np.load(output / f'{case}.npz')
        differences = {key: np.interp(time, trace['time_arr'], trace[key]) - ref[key]
                       for key in (*keys, 'q_filtered')}
        horizons = {}
        for horizon in (2., 5., 10.):
            if time[-1] < horizon / 1000 - 1e-12:
                continue
            mask = time <= horizon / 1000 + 1e-12
            horizons[str(horizon)] = {
                key: {'max_abs': float(np.max(np.abs(diff[mask]))),
                      'rmse': float(np.sqrt(np.mean(diff[mask] ** 2)))}
                for key, diff in differences.items()}
        last_ms = trace['time_arr'] >= trace['time_arr'][-1] - .001 - 1e-12
        terminal = {key: {'final': float(trace[key][-1]),
                          'minimum': float(np.min(trace[key])),
                          'maximum': float(np.max(trace[key])),
                          'last_ms_mean': float(np.mean(trace[key][last_ms]))}
                    for key in ('v_dc', 'p_filtered', 'q_filtered')}
        envelope = np.sqrt(2 / 3 * (trace['i_a'] ** 2 + trace['i_b'] ** 2 + trace['i_c'] ** 2))
        terminal['current_envelope'] = {'final': float(envelope[-1]),
                                        'maximum': float(np.max(envelope))}
        summary['cases'][case] = {'horizons_ms': horizons, 'terminal_behavior': terminal}
        if case != baseline:
            for ax, key in zip(axes, keys):
                ax.plot(time * 1000, np.maximum.accumulate(np.abs(differences[key])),
                        label=names[variant], color=colors[variant])
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes[0].legend()
    axes[-1].set_xlabel('Time [ms]')
    fig.suptitle('Cumulative maximum differences against procedural PWM')
    fig.savefig(output / 'grid_error_growth.png', dpi=150)
    plt.close(fig)
    (output / 'comparison_horizons.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = [f'# Converter comparison over {time[-1]*1000:g} ms', '',
              'The procedural PWM trajectory is the reference. Errors below are maximum absolute '
              'differences up to each stated time, using interpolation onto the reference timestamps. '
              'Integration times exclude model/solver setup and include first-use compilation.', '',
              '| Variant | Setup (s) | Integration (s) | Current error at 2 ms (pu) | '
              'At 5 ms (pu) | At 10 ms (pu) |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for case, values in summary['cases'].items():
        row = valid[case]
        errors = [f"{values['horizons_ms'][str(h)]['i_a']['max_abs']:.6g}"
                  if str(h) in values['horizons_ms'] else '—' for h in (2., 5., 10.)]
        report.append(f"| {names[case.split('_')[1]]} | {row['setup_seconds']:.2f} | "
                      f"{row['solve_seconds']:.2f} | " + ' | '.join(errors) + ' |')
    report += ['', 'Current error means phase-A current error, not a percentage.', '',
               '| Variant | Final DC voltage (pu) | Final current envelope (pu) | '
               'Final filtered P (pu) | Final filtered Q (pu) |',
               '| --- | ---: | ---: | ---: | ---: |']
    for case, values in summary['cases'].items():
        terminal = values['terminal_behavior']
        report.append(f"| {names[case.split('_')[1]]} | " + ' | '.join(
            f"{terminal[key]['final']:.6g}" for key in
            ('v_dc', 'current_envelope', 'p_filtered', 'q_filtered')) + ' |')
    report += ['', 'The current envelope is `sqrt(2/3 * (ia² + ib² + ic²))`. '
               'These are startup trajectories; numerical completion does not imply a settled response.', '',
               '[Trajectories](grid_comparison.png) · [Error growth](grid_error_growth.png) · '
               '[Timing plot](timing_comparison.png) · [Method and run status](README.md)', '',
               'Recreate this analysis from the repository root:', '', '```sh',
               'PYTHONPATH=src MPLBACKEND=Agg python '
               'src/TensyGridEngine/hybrid_boolean_mti/pwm_dq_comparison/analyze_grid_comparison.py '
               + str(output), '```', '']
    (output / 'COMPARISON.md').write_text('\n'.join(report))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    analyze(parser.parse_args().output)
