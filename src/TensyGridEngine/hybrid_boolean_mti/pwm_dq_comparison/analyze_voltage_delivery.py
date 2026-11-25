"""Time-weighted carrier averages using every accepted integration substep."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def interval_averages(time, interval_values, edges):
    """Integrate values held over each (time[k], time[k+1]] interval."""
    values = np.asarray(interval_values)
    result = []
    for start, end in zip(edges[:-1], edges[1:]):
        weights = np.maximum(0., np.minimum(time[1:], end)-np.maximum(time[:-1], start))
        if not np.isclose(weights.sum(), end-start, rtol=1e-9, atol=1e-14):
            raise ValueError('Carrier averaging interval is not fully recorded')
        result.append(np.tensordot(weights, values, axes=(0, 0))/(end-start))
    return np.asarray(result)


def analyze(output):
    data = np.load(output/'controls.npz')
    metadata = json.loads((output/'metadata.json').read_text())
    indices = {v['name']: v['index'] for v in metadata['variables']}
    time, x = data['accepted_time'], data['accepted_x']
    def signal(name):
        return x[:, indices[name]]
    command = np.column_stack([signal('v_cmd_'+axis) for axis in 'dq'])
    switched = np.column_stack([signal('v_conv_'+axis) for axis in 'dq'])
    blend = signal('switching_blend')[:, None]
    effective = blend*switched+(1-blend)*command
    # Independently check what the current state increments integrated. These
    # are the unchanged R/X/f defaults of this native grid benchmark.
    phase_current = np.column_stack([signal('i_'+phase) for phase in 'ABC'])
    bus_voltage = np.column_stack([signal('v_'+phase) for phase in 'ABC'])
    phase_switched = np.column_stack([signal('v_conv_'+phase) for phase in 'abc'])
    phase_command = np.column_stack([signal('v_ref_'+phase+'_raw') for phase in 'abc'])
    phase_effective = blend*phase_switched+(1-blend)*phase_command
    inferred_voltage = bus_voltage[1:]-.02*phase_current[1:]-(.08/(2*np.pi*50))*np.diff(
        phase_current, axis=0)/np.diff(time)[:, None]
    balance_error = float(abs(inferred_voltage-phase_effective[1:]).max())
    if balance_error > 1e-5:
        raise ValueError('Phase RL balance disagrees: check benchmark R/X/f or recorded state selection')
    # Reconstruct a left-held waveform on this recorded trajectory. This is a
    # quadrature diagnostic, not a rerun with corrected event/integration rules.
    # Scheduling rows contain the post-edge gate sequence, including when
    # accepted algebraic outputs use left limits in the deferred-edge trial.
    pwm = data['pwm']
    gate_rows = np.clip(np.searchsorted(pwm[:, 0], time[:-1], side='right')-1, 0, len(pwm)-1)
    left_gates = pwm[gate_rows, 7:10]
    theta_mid = .5*(signal('theta_pll')[:-1]+signal('theta_pll')[1:])
    scale_mid = .5*(signal('k_v_conv')[:-1]*signal('v_dc')[:-1]
                     +signal('k_v_conv')[1:]*signal('v_dc')[1:])
    phase_voltage = 2*scale_mid[:, None]*(left_gates-left_gates.mean(axis=1)[:, None])
    angles = theta_mid[:, None]+np.array([0., -2*np.pi/3, 2*np.pi/3])
    left_switched = np.column_stack([2/3*np.sum(np.sin(angles)*phase_voltage, axis=1),
                                     -2/3*np.sum(np.cos(angles)*phase_voltage, axis=1)])
    blend_mid = .5*(blend[:-1]+blend[1:])
    left_effective = blend_mid*left_switched+(1-blend_mid)*.5*(command[:-1]+command[1:])
    edges = np.arange(int(np.floor((time[-1]+1e-12)/.0002))+1)*.0002
    centers = .5*(edges[:-1]+edges[1:])
    cmd_mean = interval_averages(time, command[1:], edges)
    applied_mean = interval_averages(time, effective[1:], edges)
    left_mean = interval_averages(time, left_effective, edges)
    selected = centers >= .002
    error = applied_mean-cmd_mean
    summary = dict(converged=metadata['converged'], final_time=float(time[-1]),
                   accepted_samples=len(time), averaging_period_seconds=.0002,
                   phase_RL_voltage_balance_max_error=balance_error,
                   phase_RL_check_parameters=dict(R=.02, X=.08, frequency_hz=50.),
                   statistics_start_seconds=.002, axes={}, command_delay_sensitivity={}, windows={})
    for axis, column in zip('dq', range(2)):
        summary['axes'][axis] = dict(mean_error=float(error[selected, column].mean()),
            rmse=float(np.sqrt(np.mean(error[selected, column]**2))),
            max_abs_error=float(abs(error[selected, column]).max()),
            left_reconstruction_mean_error=float((left_mean-cmd_mean)[selected, column].mean()),
            left_reconstruction_rmse=float(np.sqrt(np.mean((left_mean-cmd_mean)[selected, column]**2))))
    for start, end in [(.002, .0085), (.0085, .01)]:
        rows = (centers >= start) & (centers < end)
        summary['windows'][f'{start*1000:g} to {end*1000:g} ms'] = dict(
            mean_applied_minus_command=error[rows].mean(axis=0).tolist(),
            mean_left_reconstruction_minus_command=(left_mean-cmd_mean)[rows].mean(axis=0).tolist())
    for delay in [0., .000025, .00005, .0001]:
        delayed_command = np.column_stack([np.interp(time-delay, time, command[:, col]) for col in range(2)])
        delayed_mean = interval_averages(time, delayed_command[1:], edges)
        summary['command_delay_sensitivity'][str(delay)] = np.sqrt(np.mean(
            (applied_mean[selected]-delayed_mean[selected])**2, axis=0)).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout='constrained')
    for col, axis in enumerate('dq'):
        axes[0, col].plot(centers*1000, cmd_mean[:, col], label='Command (carrier mean)')
        axes[0, col].plot(centers*1000, applied_mean[:, col], label='Applied (BE substep weighted)')
        axes[0, col].plot(centers*1000, left_mean[:, col], '--', label='Left-held reconstruction')
        axes[0, col].set_ylabel(axis+' voltage [pu]')
        axes[1, col].plot(centers*1000, error[:, col], label='Applied minus command')
        axes[1, col].plot(centers*1000, (left_mean-cmd_mean)[:, col], '--', label='Reconstruction minus command')
        axes[1, col].axhline(0., color='black', linewidth=.7)
        axes[1, col].set_ylabel(axis+' voltage error [pu]')
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
        ax.set_xlabel('Carrier-period center [ms]')
    fig.suptitle('Native PWM: 200 µs carrier averages; no production changes')
    fig.savefig(output/'voltage_delivery.png', dpi=160)
    plt.close(fig)
    np.savez_compressed(output/'carrier_averages.npz', time=centers, command=cmd_mean,
                        applied=applied_mean, left_reconstruction=left_mean)
    (output/'voltage_delivery.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    analyze(parser.parse_args().output)
