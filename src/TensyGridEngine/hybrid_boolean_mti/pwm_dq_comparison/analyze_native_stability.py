"""Plot and summarize a completed native procedural PWM grid trajectory."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze(output: Path):
    trace = np.load(output / 'grid_standard_StructuralCompiled.npz')
    run = json.loads((output / 'grid_standard_StructuralCompiled.json').read_text())
    t = trace['time_arr']
    currents = np.column_stack([trace['i_' + phase] for phase in 'abc'])
    envelope = np.sqrt(2 / 3 * np.sum(currents**2, axis=1))
    signals = dict(current_envelope=envelope, v_dc=trace['v_dc'],
                   p_filtered=trace['p_filtered'], q_filtered=trace['q_filtered'], q=trace['q'])
    summary = dict(run=run, signals={})
    for name, values in signals.items():
        windows = {}
        for label, start, end in [('first_ms', 0., .001), ('last_ms', t[-1]-.001, t[-1])]:
            selected = values[(t >= start - 1e-12) & (t <= end + 1e-12)]
            windows[label] = dict(min=float(selected.min()), max=float(selected.max()),
                                  mean=float(selected.mean()))
        summary['signals'][name] = dict(initial=float(values[0]), final=float(values[-1]),
                                       min=float(values.min()), max=float(values.max()), windows=windows)
    summary['q_by_ms'] = []
    for start in np.arange(0., t[-1] - 1e-12, .001):
        selected = (t >= start - 1e-12) & (t < start + .001 - 1e-12)
        summary['q_by_ms'].append(dict(start_ms=float(start*1000),
            q_mean=float(trace['q'][selected].mean()),
            q_min=float(trace['q'][selected].min()), q_max=float(trace['q'][selected].max()),
            filtered_q_mean=float(trace['q_filtered'][selected].mean())))
    summary['interpretation'] = ('Finite-horizon numerical and waveform check only; '
                                 '10 ms does not establish asymptotic stability.')
    (output / 'stability_metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for phase in 'abc':
        axes[0, 0].plot(t*1000, trace['i_' + phase], label=phase.upper(), linewidth=.8)
    axes[0, 0].set_ylabel('AC phase currents [pu]')
    axes[0, 0].legend()
    axes[0, 1].plot(t*1000, envelope)
    axes[0, 1].set_ylabel('Three-phase current envelope [pu]')
    axes[1, 0].plot(t*1000, trace['v_dc'])
    axes[1, 0].set_ylabel('DC bus voltage [pu]')
    for signal, label in [('p_filtered', 'P filtered'), ('q_filtered', 'Q filtered')]:
        axes[1, 1].plot(t*1000, trace[signal], label=label)
    axes[1, 1].set_ylabel('Filtered power [pu]')
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel('Time [ms]')
        ax.ticklabel_format(axis='y', useOffset=False)
        ax.grid(alpha=.25)
        ax.axvline(.1, color='gray', linestyle=':', linewidth=.8)
    fig.suptitle('Native three-phase procedural PWM: 5 kHz, 10 µs Backward Euler')
    fig.savefig(output / 'native_stability.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout='constrained')
    axes[0].plot(t*1000, trace['q'], linewidth=.8)
    axes[0].set_ylabel('Instantaneous Q [pu]')
    axes[1].plot(t*1000, trace['q_filtered'])
    axes[1].set_ylabel('Filtered Q [pu]')
    for ax in axes:
        ax.axhline(0., color='black', linestyle='--', linewidth=.8, label='Q reference')
        ax.axvline(.1, color='gray', linestyle=':', linewidth=.8)
        ax.grid(alpha=.25)
        ax.legend()
    axes[1].set_xlabel('Time [ms]')
    fig.suptitle('Native procedural PWM: reactive power at zero Q reference')
    fig.savefig(output / 'native_reactive_power.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    analyze(parser.parse_args().output)
