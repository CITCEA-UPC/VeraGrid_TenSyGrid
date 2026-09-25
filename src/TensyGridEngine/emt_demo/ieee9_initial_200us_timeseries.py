"""Plot the first 0.2 ms of the initialized nonlinear IEEE9 EMT case."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from VeraGridEngine.Simulations.EMT.solvers.StructuralVectorizedSolver import (
    StructuralVectorizedSolver,
)
from VeraGridEngine.enumerations import DynamicIntegrationMethod

from ieee9_common import build_ieee9_case, signal


DURATION = 0.2e-3
TIME_STEP = 2.5e-6
OUTPUT = Path(__file__).with_name("ieee9_initial_200us_timeseries.png")
DATA = Path(__file__).with_name("ieee9_initial_200us_timeseries.npz")


def main() -> None:
    case = build_ieee9_case(
        multilinear=False,
        time_step=TIME_STEP,
        simulation_time=DURATION,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
    )
    solver = StructuralVectorizedSolver(
        problem=case.problem,
        t0=0.0,
        t_end=DURATION,
        h=TIME_STEP,
        method=DynamicIntegrationMethod.DaeTrapezoidal,
        verbose=False,
    )
    time, values, _derivatives, initialized, converged = solver.simulate()
    if not initialized or not converged:
        raise RuntimeError(
            f"IEEE9 EMT simulation failed: initialized={initialized}, converged={converged}"
        )

    traces = {
        "omega": signal(case, values, "omega"),
        "P": signal(case, values, "P"),
        "Q": signal(case, values, "Q"),
        "V": signal(case, values, "V"),
        "i_g_A": signal(case, values, "i_g_A"),
        "v_f_A": signal(case, values, "v_f_A"),
    }
    if DURATION >= 1.0e-3:
        time_plot = np.asarray(time) * 1e3
        time_label = "time [milliseconds]"
    else:
        time_plot = np.asarray(time) * 1e6
        time_label = "time [microseconds]"
    np.savez(DATA, time=time_plot, **traces)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    specifications = (
        ("omega", "GFM frequency", "omega [pu]", "tab:blue"),
        ("P", "GFM active power", "P [pu]", "tab:red"),
        ("Q", "GFM reactive power", "Q [pu]", "tab:orange"),
        ("V", "GFM voltage command", "V [pu]", "tab:green"),
        ("i_g_A", "GFM grid-side phase-A current", "i_g,A [pu]", "tab:purple"),
        ("v_f_A", "GFM filter phase-A voltage", "v_f,A [pu]", "tab:brown"),
    )
    for axis, (key, title, ylabel, color) in zip(axes.flat, specifications):
        axis.plot(time_plot, traces[key], color=color, linewidth=1.5)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(alpha=0.3)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    for axis in axes[-1, :]:
        axis.set_xlabel(time_label)
    fig.suptitle(f"Initialized IEEE9 EMT response during the first {DURATION * 1e3:g} ms")
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)

    print(f"plot={OUTPUT}")
    for name, trace in traces.items():
        print(
            f"{name}: initial={trace[0]:+.9e}, final={trace[-1]:+.9e}, "
            f"max_delta={np.max(np.abs(trace - trace[0])):.9e}"
        )


if __name__ == "__main__":
    main()
