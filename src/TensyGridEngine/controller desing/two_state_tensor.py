# SPDX-License-Identifier: MPL-2.0
"""Unrestricted two-state multilinear controller, evaluated by native VeraGrid."""
from pathlib import Path
import json
import argparse
import numpy as np
from scipy.optimize import differential_evolution
from cp_governor import (SymbolicCPTensor, block_from_cp, Const, Parameters,
                         run_study)

# Every nonconstant multilinear feature in three independent variables.
FEATURES = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
FEATURE_NAMES = ['x1', 'x2', 'e', 'x1*x2', 'x1*e', 'x2*e', 'x1*x2*e']
SEED = np.array([[-5, 0, 5, 0, 0, 0, 0],
                 [2, -2, 0, 0, 0, 0, 0],
                 [0, .1, .03, 0, 0, 0, 0]], dtype=float)


def factory(coefficients):
    C = np.asarray(coefficients, dtype=float)
    if C.shape != (3, 7) or not np.isfinite(C).all():
        raise ValueError('Expected a finite 3 by 7 coefficient matrix')
    def build(vf, parameters):
        x1, x2 = [vf.add_var(n) for n in ('tensor_x1', 'tensor_x2')]
        omega = vf.add_var('omega_in', shared_reference='omega_reference')
        te = vf.add_var('Te_init', shared_reference='te_reference')
        tm = vf.add_var('Tm_gov', shared_reference='tm_reference')
        tref = vf.add_var('Tref')
        tensor = SymbolicCPTensor.empty(3, 3, 8)
        tensor.output_factor[:, :7] = C
        for r, modes in enumerate(FEATURES):
            for mode in modes:
                tensor.factors[mode][:, r] = [200, -200] if mode == 2 else [0, 1]
        tensor.output_factor[2, 7] = tref
        return block_from_cp(name='Full two-state CP controller', n_states=2,
            states=[x1, x2], inputs=[omega, te], outputs=[tm],
            tensor_variables=[x1, x2, omega], tensor=tensor,
            parameters={tref: te}, initial_values={x1: Const(0), x2: Const(0), tm: te})
    return build


def evaluate(C, step=.05, dt=.01):
    study = run_study(step=step, duration=30, dt=dt, controller_factory=factory(C))
    _, machine, ctrl, _, modal, rms = study
    res = rms.results
    t = (res.time_array-res.time_array[0]).total_seconds().to_numpy()
    def series(v):
        return res.values[:, res.uid2idx[v.uid], 0]
    x, y = [series(v) for v in ctrl.state_vars]
    omega = series(machine.out_vars[2])
    torque = series(ctrl.out_vars[0])
    deviation = 50*(omega-1)
    rate = np.diff(torque)/np.diff(t)
    modes = modal.results.eigenvalues
    modes = modes[np.abs(modes) > 1e-7]
    initial = modal.problem.get_x0()
    dx = modal.problem.get_dx0()
    residual = max(np.linalg.norm(modal.problem.rhs_state(initial, dx), np.inf),
                   np.linalg.norm(modal.problem.rhs_algebraic(initial, dx), np.inf))
    metrics = dict(peak_hz=float(abs(deviation[t >= 2]).max()),
        final_error_hz=float(abs(deviation[-1])),
        late_oscillation_hz=float(np.ptp(deviation[t >= 25])),
        torque_min=float(torque.min()), torque_max=float(torque.max()),
        max_torque_rate=float(abs(rate).max()), max_state=float(max(abs(x).max(), abs(y).max())),
        max_mode_real=float(modes.real.max()), initial_residual=float(residual))
    violation = max(0, -metrics['torque_min'], metrics['torque_max']-1.1,
        metrics['max_torque_rate']-.4, metrics['max_state']-3,
        metrics['max_mode_real']+.05, metrics['late_oscillation_hz']-.01,
        metrics['final_error_hz']-max(.001, .15*abs(step)/.05))
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError('Nonfinite trajectory')
    metrics.update(violation=float(violation), feasible=bool(violation <= 1e-9))
    return metrics, np.column_stack([t, deviation, x, y, torque])


def run(output, iterations):
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = [float('inf'), None]
    # All 21 entries vary: no state equation or output is frozen.
    scales = np.array([[2, 1, 3, .5, .5, .5, .3],
                       [1, 1, 1, .5, .5, .5, .3],
                       [.04, .04, .04, .02, .02, .02, .01]])
    def objective(z):
        C = SEED + np.asarray(z).reshape(3, 7)*scales
        try:
            m, curve = evaluate(C)
            violation = m['violation']
            if violation == 0:
                for step in (-.05, .075):
                    check, _ = evaluate(C, step)
                    violation = max(violation, check['violation'])
                    if step < 0:
                        violation = max(violation, check['peak_hz']-.25)
                    if violation > 0:
                        break
            score = m['peak_hz'] if violation <= 1e-9 else 10+100*violation
            m.update(feasible=bool(violation <= 1e-9), violation=float(violation))
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            m, score = dict(feasible=False, error=str(exc)), 1e6
        history.append(dict(C=C.tolist(), score=float(score), **m))
        if m['feasible'] and score < best[0]:
            best[:] = [score, C.copy()]
            print(f'Candidate {len(history)}: peak {score:.6f} Hz', flush=True)
            (output/'incumbent.json').write_text(json.dumps(history[-1], indent=2))
        (output/'search_history.json').write_text(json.dumps(history, indent=2))
        return score
    rng = np.random.default_rng(23)
    initial = rng.uniform(-.6, .6, (24, 21))
    initial[0] = 0
    differential_evolution(objective, [(-1, 1)]*21, init=initial,
        maxiter=iterations, seed=23, polish=False, tol=.001)
    if best[1] is None:
        raise RuntimeError('No feasible controller')
    C = best[1]
    (output/'controller.json').write_text(json.dumps(dict(C=C.tolist(),
        features=FEATURE_NAMES, states=2, cp_components=8, error_scale=.005), indent=2))
    (output/'search_config.json').write_text(json.dumps(dict(seed=23, iterations=iterations,
        population=24, seed_matrix=SEED.tolist(), scales=scales.tolist(),
        normalized_bounds=[-1, 1], search_dt=.01, duration=30,
        target=.05, constraint_steps=[-.05, .075]), indent=2))
    compare(C, output)


def compare(C, output):
    from tune_tensor_governor import evaluate as old_evaluate
    previous = Parameters(c_ev=.08, c_et=-.08, c_evt=-.01, c_vt=-.001)
    rows, curves = [], {}
    for step in (.05, -.05, .025, .075):
        for label, p in [('baseline', Parameters()), ('previous tensor', previous), ('full two-state tensor', None)]:
            if p is None:
                metrics, data = evaluate(C, step, .005)
                if step < 0 and metrics['peak_hz'] > .25:
                    metrics['feasible'] = False
            else:
                metrics, raw = old_evaluate(p, step=step, dt=.005)
                data = raw[:, [0, 1, 2, 3, 3]]
            rows.append(dict(controller=label, step=step, **metrics))
            if step == .05:
                curves[label] = data
            print(label, step, metrics['peak_hz'], metrics['feasible'], flush=True)
    ablations = []
    for label, matrix in [('linear seed', SEED), ('selected linear part', np.column_stack((C[:, :3], np.zeros((3, 4)))))]:
        metrics, data = evaluate(matrix, .05, .005)
        ablations.append(dict(controller=label, **metrics))
    (output/'ablation.json').write_text(json.dumps(ablations, indent=2))
    (output/'comparison.json').write_text(json.dumps(rows, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout='constrained')
    for label, data in curves.items():
        ax[0].plot(data[:, 0], data[:, 1], label=label)
        ax[1].plot(data[:, 0], data[:, 4], label=label)
    data = curves['full two-state tensor']
    ax[2].plot(data[:, 0], data[:, 2], label='x1')
    ax[2].plot(data[:, 0], data[:, 3], label='x2')
    for a, title in zip(ax, ['Frequency deviation [Hz]', 'Mechanical torque [pu]', 'Internal states']):
        a.set_ylabel(title)
        a.grid(alpha=.25)
        a.legend()
        a.axvline(2, color='gray', ls=':')
    ax[2].set_xlabel('Time [s]')
    fig.suptitle('Two states: CP terms in both dynamics and torque output | +5 MW')
    fig.savefig(output/'comparison.png', dpi=150)
    plt.close(fig)
    np.savetxt(output/'response.csv', data, delimiter=',', header='time_s,deviation_hz,x1,x2,torque_pu', comments='')
    if not all(r['feasible'] for r in rows if r['controller'] == 'full two-state tensor'):
        raise RuntimeError('Fine-step verification failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'two_state_results')
    args = parser.parse_args()
    run(args.output, args.iterations)
