"""Compare native converter trajectories when only current feedback filtering changes."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compare(baseline, experiment, output):
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, layout='constrained')
    metrics = {}
    deferred_trial = json.loads((experiment/'metadata.json').read_text()).get('defer_pwm_edges', False)
    common_end = min(float(np.load(directory/'controls.npz')['time'][-1]) for directory in (baseline, experiment))
    for directory in (baseline, experiment):
        metadata = json.loads((directory/'metadata.json').read_text())
        tau = metadata.get('current_filter_seconds_at_5khz', .001)
        label = f'{tau*1e6:g} µs current filter'
        if metadata.get('defer_pwm_edges', False):
            label += ', deferred edges'
            deferred_trial = True
        if deferred_trial:
            label = 'Deferred edge ordering' if directory == experiment else 'Original edge ordering'
        data = np.load(directory/'controls.npz')
        t, x = data['time'], data['x']
        indices = {v['name']: v['index'] for v in metadata['variables']}
        def signal(name):
            return x[:, indices[name]]
        selected = (t >= .0005) & (t <= common_end+1e-12)
        last_ms = t >= t[-1]-.001-1e-12
        metrics[label] = dict(initialized=metadata['initialized'], converged=metadata['converged'],
            final_time=float(t[-1]), vdc_min=float(signal('v_dc').min()), vdc_final=float(signal('v_dc')[-1]),
            tracking_metrics_end_seconds=common_end,
            q_filtered_min=float(signal('Q_f').min()), q_filtered_max=float(signal('Q_f').max()),
            q_filtered_final=float(signal('Q_f')[-1]), q_filtered_final_available_ms_mean=float(signal('Q_f')[last_ms].mean()),
            id_tracking_rmse=float(np.sqrt(np.mean((signal('i_d_ref')[selected]-signal('i_d')[selected])**2))),
            iq_tracking_rmse=float(np.sqrt(np.mean((signal('i_q_ref')[selected]-signal('i_q')[selected])**2))),
            id_final=float(signal('i_d')[-1]), id_reference_final=float(signal('i_d_ref')[-1]),
            solve_seconds=metadata['solve_seconds'])
        for ax, name in zip(axes.flat, ['i_d', 'i_q', 'v_dc', 'Q_f', 'i_d_meas_f', 'i_q_meas_f']):
            ax.plot(t*1000, signal(name), label=label, linewidth=.9)
            ax.set_ylabel(name+' [pu]')
            if not metadata['converged']:
                ax.axvline(t[-1]*1000, color='red', linestyle=':', linewidth=.8)
        for ax, name in [(axes[0, 0], 'i_d_ref'), (axes[0, 1], 'i_q_ref')]:
            ax.plot(t*1000, signal(name), linestyle='--', linewidth=.8, label=label+' reference')
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.set_xlabel('Time [ms]')
        ax.legend(fontsize=7)
        ax.ticklabel_format(axis='y', useOffset=False)
    fig.suptitle('Native procedural PWM: edge ordering changed; gains and filters unchanged'
                 if deferred_trial else 'Native procedural PWM: only current-feedback filtering changed; gains unchanged')
    fig.savefig(output/'filter_comparison.png', dpi=160)
    plt.close(fig)
    (output/'filter_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    compare(args.baseline, args.experiment, args.output)
