"""Test whether the IRPu-to-e'_q path causes the IEEE9 initial transient."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ieee9_emt_damping_ablation as ablation
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block


TIME_STEP = 5.0e-6
SIMULATION_TIME = 0.04


def signal(problem, values, model, name: str) -> np.ndarray:
    variable = find_name_in_block(name, model)
    if variable is None:
        raise RuntimeError(f"Could not find {name} in generator model")
    return values[:, int(problem.get_var_idx(variable))]


def main() -> None:
    ablation.TIME_STEP = TIME_STEP
    ablation.SIMULATION_TIME = SIMULATION_TIME
    problem, time, values = ablation.run_case(damping=0.0, freeze_e_qp=True)
    omega, te, tm = ablation.machine_signals(problem, values)
    model = next(gen.emt_model for gen in problem.grid.generators if not gen.bus.is_slack)

    irpu = signal(problem, values, model, "IRPu")
    e_qp = signal(problem, values, model, "e_qp_")
    i_d = signal(problem, values, model, "i_d_")
    i_q = signal(problem, values, model, "i_q_")

    for sample_time in (0.0, TIME_STEP, 1.0e-3, 20.0e-3, 40.0e-3):
        idx = int(np.argmin(np.abs(time - sample_time)))
        print(
            f"t={time[idx]:.9e}, omega={omega[idx]:.12f}, "
            f"Te={te[idx]:.12f}, Tm={tm[idx]:.12f}, "
            f"Te_minus_Tm={te[idx] - tm[idx]:+.12f}, "
            f"IRPu={irpu[idx]:.12f}, e_qp={e_qp[idx]:.12f}, "
            f"i_d={i_d[idx]:.12f}, i_q={i_q[idx]:.12f}"
        )

    print(
        f"changes: delta_omega={omega[-1] - omega[0]:+.12e}, "
        f"delta_Te={te[-1] - te[0]:+.12e}, "
        f"delta_IRPu={irpu[-1] - irpu[0]:+.12e}, "
        f"delta_e_qp={e_qp[-1] - e_qp[0]:+.12e}, "
        f"delta_i_d={i_d[-1] - i_d[0]:+.12e}, "
        f"delta_i_q={i_q[-1] - i_q[0]:+.12e}"
    )

    time_ms = 1.0e3 * time
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 10.0), sharex=True, constrained_layout=True)
    axes[0].plot(time_ms, omega)
    axes[0].set_ylabel("omega (p.u.)")
    axes[1].plot(time_ms, te, label="Te")
    axes[1].plot(time_ms, tm, "--", label="Tm")
    axes[1].set_ylabel("Torque (p.u.)")
    axes[1].legend()
    axes[2].plot(time_ms, irpu - irpu[0], label="delta IRPu")
    axes[2].plot(time_ms, e_qp - e_qp[0], label="delta e'_q")
    axes[2].set_ylabel("Field delta (p.u.)")
    axes[2].legend()
    axes[3].plot(time_ms, i_d - i_d[0], label="delta id")
    axes[3].plot(time_ms, i_q - i_q[0], label="delta iq")
    axes[3].set_ylabel("Current delta (p.u.)")
    axes[3].set_xlabel("Time (ms)")
    axes[3].legend()
    axes[0].set_title("IEEE9 EMT IRPu causality test — e'_q frozen")
    for axis in axes:
        axis.grid(True, alpha=0.3)

    output = Path(__file__).with_name("ieee9_emt_irpu_causality_frozen_eqp.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"plot={output}")


if __name__ == "__main__":
    main()
