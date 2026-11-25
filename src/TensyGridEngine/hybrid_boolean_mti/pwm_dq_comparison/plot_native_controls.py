"""Analyze a saved native-control diagnostic without rerunning the grid."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze(output):
    metadata = json.loads((output/'metadata.json').read_text())
    data = np.load(output/'controls.npz')
    t, x, pwm = data['time'], data['x'], data['pwm']
    indices = {v['name']: v['index'] for v in metadata['variables']}
    def signal(name):
        return x[:, indices[name]]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, layout='constrained')
    for axis, direction in zip(axes[0], 'dq'):
        for name, label in [(f'i_{direction}_ref', 'Reference'), (f'i_{direction}', 'Actual'),
                            (f'i_{direction}_meas_f', 'Filtered measurement')]:
            axis.plot(t*1000, signal(name), label=label, linewidth=.9)
        axis.set_ylabel(f'{direction}-axis current [pu]')
    for axis, direction in zip(axes[1], 'dq'):
        for suffix, label in [('_u', 'Before limit'), ('', 'After limit')]:
            axis.plot(t*1000, signal(f'v_cmd_{direction}{suffix}'), label=label, linewidth=1)
        axis.set_ylabel(f'{direction}-axis voltage command [pu]')
    for phase in 'abc':
        axes[2, 0].plot(t*1000, signal('m_'+phase), label=phase.upper(), linewidth=.8)
    axes[2, 0].set_ylabel('Modulation command')
    for name in ['P', 'P_f', 'Q_f']:
        axes[2, 1].plot(t*1000, signal(name), label=name)
    axes[2, 1].set_ylabel('Power [pu]')
    for axis in axes.flat:
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
        axis.set_xlabel('Time [ms]')
    fig.savefig(output/'current_and_voltage_commands.png', dpi=150)
    plt.close(fig)
    active = pwm[:, 1] > 1.
    stale = active & (pwm[:, 3] <= pwm[:, 0] + 1e-12)
    summary = dict(stale_schedule_calls=int(stale.sum()),
                   stale_times_ms=(pwm[stale, 0]*1000).tolist(), signals={})
    for name in ['i_d_ref', 'i_d', 'i_q_ref', 'i_q', 'v_cmd_d_u', 'v_cmd_d', 'v_cmd_q_u', 'v_cmd_q',
                 'v_lim', 'omega_pll', 'theta_pll', 'v_dc', 'i_dc_conv']:
        values = signal(name)
        summary['signals'][name] = dict(min=float(values.min()), max=float(values.max()), final=float(values[-1]))
    for direction in 'dq':
        delta = signal(f'v_cmd_{direction}_u')-signal(f'v_cmd_{direction}')
        summary[f'{direction}_voltage_saturated_samples'] = int((abs(delta)>1e-7).sum())
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), layout='constrained')
    for col, (start, end) in enumerate([(1.8, 2.3), (8.7, 9.2)]):
        selected = (pwm[:, 0]*1000>=start)&(pwm[:, 0]*1000<=end)
        rows = pwm[selected]
        for phase in range(3):
            axes[0, col].step(rows[:, 0]*1000, rows[:, 7+phase]+1.2*phase, where='post', label='ABC'[phase])
        axes[1, col].plot(rows[:, 0]*1000, (rows[:, 3]-rows[:, 0])*1e6, '.-', linewidth=.7)
        axes[1, col].axhline(0., color='red', linestyle='--')
        selected = (t*1000>=start)&(t*1000<=end)
        for direction in 'dq':
            axes[2, col].plot(t[selected]*1000, signal('i_'+direction)[selected], label=direction)
        axes[0, col].set_ylabel('Gate states (offset)')
        axes[1, col].set_ylabel('Scheduled interval end minus time [µs]')
        axes[2, col].set_ylabel('Actual dq currents [pu]')
        axes[0, col].legend()
        axes[2, col].legend()
    for axis in axes.flat:
        axis.set_xlabel('Time [ms]')
        axis.grid(alpha=.25)
    fig.savefig(output/'pwm_timing_zoom.png', dpi=150)
    plt.close(fig)
    (output/'diagnosis.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    analyze(parser.parse_args().output)
