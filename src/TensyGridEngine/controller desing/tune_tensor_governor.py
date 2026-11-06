# SPDX-License-Identifier: MPL-2.0
"""Tune multilinear CP interaction coefficients using native VeraGrid studies."""
from dataclasses import replace, asdict
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import differential_evolution
from cp_governor import Parameters, run_study

GAIN_NAMES = ('c_ev', 'c_et', 'c_evt', 'c_vt')


def measure(study, parameters, event_time=2., final_error_limit=.15):
    grid, machine, controller, pf, modal, rms = study
    results = rms.results
    t = (results.time_array-results.time_array[0]).total_seconds().to_numpy()
    def series(var):
        return results.values[:, results.uid2idx[var.uid], 0]
    omega = series(machine.out_vars[2])
    v, tau = [series(var) for var in controller.state_vars]
    tref = v[0]
    e, a, b = (1-omega)/.005, (v-tref)/.1, (tau-tref)/.1
    drive = (tref - parameters.K*(omega-1)*(1+parameters.alpha*(v-tref))
             + parameters.c_ev*e*a + parameters.c_et*e*b
             + parameters.c_evt*e*a*b + parameters.c_vt*a*b)
    rate = (drive-v)/parameters.Tg
    deviation = 50*(omega-1)
    if not np.all(np.isfinite(np.column_stack((omega, v, tau, drive, rate)))):
        raise ValueError('Nonfinite controller trajectory')
    post = t >= event_time
    modes = modal.results.eigenvalues
    dynamic_modes = modes[np.abs(modes) > 1e-7]
    metrics = dict(peak_hz=float(np.max(np.abs(deviation[post]))),
                   final_error_hz=float(abs(deviation[-1])),
                   final_error_limit_hz=final_error_limit,
                   late_oscillation_hz=float(np.ptp(deviation[t >= t[-1]-5.])),
                   valve_min=float(v.min()), valve_max=float(v.max()),
                   torque_min=float(tau.min()), torque_max=float(tau.max()),
                   command_min=float(drive.min()), command_max=float(drive.max()),
                   max_valve_rate=float(np.max(np.abs(rate))),
                   max_mode_real=float(dynamic_modes.real.max()))
    # Illustrative admissibility envelope, checked on simulated trajectories.
    violation = max(0., -metrics['valve_min'], metrics['valve_max']-1.1,
                    -metrics['torque_min'], metrics['torque_max']-1.1,
                    -metrics['command_min'], metrics['command_max']-1.2,
                    metrics['max_valve_rate']-.5,
                    metrics['final_error_hz']-final_error_limit, metrics['max_mode_real']+.05,
                    metrics['late_oscillation_hz']-.01)
    metrics['feasible'] = bool(violation <= 1e-9)
    metrics['violation'] = float(violation)
    return metrics, np.column_stack((t, deviation, v, tau, drive, rate))


def evaluate(p, step=.05, dt=.01, duration=30.):
    return measure(run_study(p, step=step, dt=dt, duration=duration), p,
                   final_error_limit=max(.001, .15*abs(step)/.05))


def tune(output, iterations=8):
    output.mkdir(parents=True, exist_ok=True)
    baseline = Parameters()
    baseline_metrics, _ = evaluate(baseline)
    history = []
    best = {'score': float('inf'), 'gains': None}
    def objective(gains):
        p = replace(baseline, **dict(zip(GAIN_NAMES, map(float, gains))))
        try:
            metrics, trajectory = evaluate(p)
            # Opposite-sign and larger disturbances are now explicit tuning constraints.
            # A rate margin reduces acceptance of coarse-step numerical damping artifacts.
            extra_violation = max(0., metrics['max_valve_rate']-.4)
            if metrics['feasible'] and extra_violation == 0:
                for constraint_step in (-.05, .075):
                    check, _ = evaluate(p, step=constraint_step)
                    extra_violation = max(extra_violation, check['violation'], check['max_valve_rate']-.4)
                    if constraint_step < 0:
                        extra_violation = max(extra_violation, check['peak_hz']-.25)
                    if extra_violation > 0:
                        break
            metrics['violation'] = max(metrics['violation'], extra_violation)
            metrics['feasible'] = bool(metrics['violation'] <= 1e-9)
            score = metrics['peak_hz'] if metrics['feasible'] else 10 + 100*metrics['violation']
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            metrics = {'feasible': False, 'error': str(exc)}
            score = 1e6
        history.append(dict(gains=list(map(float, gains)), score=float(score), **metrics))
        if metrics['feasible'] and score < best['score']:
            best.update(score=float(score), gains=np.array(gains))
            print(f"Evaluation {len(history)}: peak {score:.6f} Hz, gains {gains}", flush=True)
            np.savetxt(output/'incumbent_response.csv', trajectory, delimiter=',',
                       header='time_s,frequency_deviation_hz,valve,torque,command,valve_rate', comments='')
            (output/'incumbent.json').write_text(json.dumps(history[-1], indent=2)+'\n')
        if len(history) % 20 == 0:
            (output/'search_history.json').write_text(json.dumps(history, indent=2)+'\n')
            print(f"Completed {len(history)} candidates; best feasible peak {best['score']}", flush=True)
        return score
    # Seeded bounded search; no change to inertia, lag constants or proportional gain.
    bounds = [(-.12, .12), (-.12, .12), (-.06, .06), (-.015, .015)]
    rng = np.random.default_rng(19)
    initial = np.array([[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(20)])
    for i, gain in enumerate((0., .02, .04, .06, .08)):
        initial[i] = [gain, -gain, -.01, -.001]
    differential_evolution(objective, bounds, maxiter=iterations, init=initial,
                           seed=19, x0=np.zeros(4), polish=False, tol=.002,
                           workers=1, updating='immediate')
    if best['gains'] is None:
        raise RuntimeError('No feasible candidate found; inspect search_history.json')
    p = replace(baseline, **dict(zip(GAIN_NAMES, map(float, best['gains']))))
    (output/'parameters.json').write_text(json.dumps(asdict(p), indent=2)+'\n')
    (output/'search_history.json').write_text(json.dumps(history, indent=2)+'\n')
    (output/'search_config.json').write_text(json.dumps(dict(
        bounds=bounds, seed=19, iterations=iterations, target_step=.05,
        constraint_steps=[-.05, .075], search_dt=.01, duration=30.), indent=2)+'\n')
    compare(output, p)


def compare(output, p):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = {}
    # Fine-step target verification plus independent step sizes/sign.
    for step in (.05, .025, -.05, .075):
        for label, candidate in (('baseline', Parameters()), ('tensor', p)):
            metrics, trajectory = evaluate(candidate, step=step, dt=.005, duration=30.)
            if step < 0 and metrics['peak_hz'] > .25:
                metrics['feasible'] = False
                metrics['violation'] = max(metrics['violation'], metrics['peak_hz']-.25)
            rows.append(dict(controller=label, step_pu=step, **metrics))
            if step == .05:
                curves[label] = trajectory
                np.savetxt(output/f'{label}_response.csv', trajectory, delimiter=',',
                           header='time_s,frequency_deviation_hz,valve,torque,command,valve_rate', comments='')
            print(label, step, metrics, flush=True)
    (output/'comparison.json').write_text(json.dumps(rows, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout='constrained')
    for label, data in curves.items():
        axes[0].plot(data[:, 0], data[:, 1], label=label)
        axes[1].plot(data[:, 0], data[:, 2], label=label)
        axes[2].plot(data[:, 0], data[:, 5], label=label)
    axes[0].set_ylabel('Frequency deviation [Hz]')
    axes[1].set_ylabel('Valve [pu]')
    axes[2].set(ylabel='Valve rate [pu/s]', xlabel='Time [s]')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
        ax.axvline(2., color='gray', linestyle=':')
    fig.suptitle('VeraGrid RMS DAE: added multilinear CP interactions | +5 MW load')
    fig.savefig(output/'comparison.png', dpi=160)
    plt.close(fig)
    verified = all(row['feasible'] for row in rows if row['controller'] == 'tensor')
    target = [row for row in rows if row['step_pu'] == .05]
    improvement = 100*(1-target[1]['peak_hz']/target[0]['peak_hz'])
    (output/'verification.json').write_text(json.dumps(dict(
        all_tensor_scenarios_feasible=verified, target_peak_reduction_percent=improvement,
        dt=.005, duration=30.), indent=2)+'\n')
    if not verified or improvement <= 0:
        raise RuntimeError('Selected candidate failed fine-step verification; inspect comparison.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'tensor_tuning_final')
    parser.add_argument('--parameters', type=Path, help='Verify saved parameters without tuning')
    args = parser.parse_args()
    if args.parameters:
        compare(args.output, Parameters(**json.loads(args.parameters.read_text())))
    else:
        tune(args.output, args.iterations)
