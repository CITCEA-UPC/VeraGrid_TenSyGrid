"""Check IEEE-9 startup against a high-accuracy ODE reference."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from TensyGridEngine.emt.emt_explicit_converter_demo.experiments.ieee9_pi_charge_event_timeseries import compile_terms, polynomial_rhs
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_pi_charge_native import build_native_problem
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.ieee9_pi_charge_polynomial import extract


def main():
    full, polynomial, dynamic, charges, Q, _ = extract(replacement_B=0.1)
    n, m = len(full.get_state_vars()), len(full.get_algebraic_vars())
    x0 = full.get_x0()
    z0 = np.r_[x0[dynamic], x0[n + m - 27:n + m]]
    charge_error = np.linalg.norm(x0[charges] - Q @ z0[len(dynamic):])
    if charge_error > 1e-9:
        raise RuntimeError(f"Initial charge inconsistency: {charge_error:g}")

    end = 0.01
    traces = {}
    for label, step in (("native 100 µs", 1e-4), ("native 25 µs", 2.5e-5)):
        native = build_native_problem(polynomial, z0, step=step)
        result = native.simulate(t0=0.0, t_end=end)
        if not result.initialized or not result.converged:
            raise RuntimeError(f"{label} native simulation failed")
        voltage = result.values[:, len(dynamic):].reshape(-1, 9, 3)
        voltage_rms = np.sqrt(np.mean(voltage[:, [1, 2, 5]] ** 2, axis=2))
        traces[label] = (result.time, voltage_rms)

    terms = compile_terms(polynomial)
    reference = solve_ivp(lambda _, z: polynomial_rhs(z, *terms),
                          (0.0, end), z0,
                          t_eval=traces["native 100 µs"][0], method="BDF",
                          rtol=1e-9, atol=1e-11, max_step=2.5e-5)
    if not reference.success:
        raise RuntimeError(reference.message)
    voltage = reference.y.T[:, len(dynamic):].reshape(-1, 9, 3)
    traces["BDF reference"] = (reference.t,
                                np.sqrt(np.mean(voltage[:, [1, 2, 5]] ** 2, axis=2)))

    for label, (_, rms) in traces.items():
        print(f"{label}: initial={rms[0]}, minimum={rms.min(axis=0)}, final={rms[-1]}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             constrained_layout=True)
    for j, bus in enumerate((1, 2, 5)):
        for label, (t, rms) in traces.items():
            axes[j].plot(t * 1e3, rms[:, j], label=label)
        axes[j].set_ylabel(f"Bus {bus} RMS (pu)")
        axes[j].grid(alpha=0.3)
        axes[j].legend()
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Modified IEEE-9 startup: corrected source initialization")
    path = Path(__file__).resolve().parents[1] / "simulation_data" / "experiments" / "ieee9_pi_charge_init_check" / "comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    print(f"plot={path}")


if __name__ == "__main__":
    main()
