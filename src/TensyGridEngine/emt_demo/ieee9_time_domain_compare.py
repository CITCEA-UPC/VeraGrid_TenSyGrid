"""Compare nonlinear and multilinear IEEE9 EMT simulations for 0.2 seconds."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver

from ieee9_common import build_ieee9_case, signal


TIME_STEP = 2.5e-6
SIMULATION_TIME = 0.2
OUTPUT = Path(__file__).with_name("ieee9_time_domain_compare.png")


def simulate(multilinear: bool):
    """Build and simulate one independent IEEE9 formulation."""
    case = build_ieee9_case(multilinear, TIME_STEP, SIMULATION_TIME)
    solver = build_emt_solver(
        options=case.problem.options,
        problem=case.problem,
        t0=0.0,
        t_end=SIMULATION_TIME,
        h=TIME_STEP,
        method=case.problem.options.integration_method,
    )
    time, values, _derivatives, initialized, converged = solver.simulate(
        boundary_updater=case.problem
    )
    if not initialized or not converged:
        raise RuntimeError(f"EMT simulation failed: initialized={initialized}, converged={converged}")
    return case, np.asarray(time), np.asarray(values)


def main() -> None:
    reference, t_ref, y_ref = simulate(multilinear=False)
    multilinear, t_ml, y_ml = simulate(multilinear=True)
    print(f"reference problem:   {type(reference.problem).__name__}")
    print(f"multilinear problem: {type(multilinear.problem).__name__}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, ("omega", "P", "Q")):
        ref = signal(reference, y_ref, name)
        ml = signal(multilinear, y_ml, name)
        axis.plot(t_ref, ref, label="nonlinear reference")
        axis.plot(t_ml, ml, "--", label="multilinear")
        axis.set_ylabel(f"{name} (pu)")
        axis.grid(alpha=0.3)
        axis.legend()
        print(f"max |ML-reference| {name}: {np.max(np.abs(ml - ref)):.6e}")
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("IEEE9 EMT: nonlinear versus multilinear GFM")
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(f"plot={OUTPUT}")


if __name__ == "__main__":
    main()

