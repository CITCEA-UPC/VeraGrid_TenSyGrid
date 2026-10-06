"""Validate the IEEE9 GFM mode with a direct small time-domain perturbation."""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt import ieee9_emt_ibr_simulation as ieee9
from TensyGridEngine.emt.emt_ieee_veragrid_demo.ieee9_common import build_ieee9_case, signal


TIME_STEP = 1.0e-5
SIMULATION_TIME = float(os.environ.get("IEEE9_GFM_EVENT_TIME", "1.0"))
STEP_TIME = 0.02
STEP_SIZE = 1.0e-3
OUTPUT = Path(__file__).with_name("ieee9_emt_gfm_event_validation.png")


def main() -> None:
    case = build_ieee9_case(False, TIME_STEP, SIMULATION_TIME)
    validation = ieee9._load_deliverable_models()[2]
    time, values, _derivatives, initialized, converged = validation.run_emt_with_p_ref_step(
        case.problem,
        case.problem.options,
        case.gfm_model,
        step_time=STEP_TIME,
        step_delta=STEP_SIZE,
    )
    if not initialized or not converged:
        raise RuntimeError(f"EMT event failed: initialized={initialized}, converged={converged}")

    omega = signal(case, values, "omega")
    power = signal(case, values, "P")
    reactive = signal(case, values, "Q")
    post = time >= STEP_TIME
    for name, trace in (("omega", omega), ("P", power), ("Q", reactive)):
        initial = float(np.mean(trace[time < STEP_TIME]))
        deviation = np.abs(trace[post] - initial)
        midpoint = max(1, deviation.size // 2)
        first_peak = float(np.max(deviation[:midpoint]))
        second_peak = float(np.max(deviation[midpoint:]))
        print(f"{name}: first_half_peak={first_peak:.9e}, second_half_peak={second_peak:.9e}")

    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for axis, trace, label in zip(axes, (omega, power, reactive), ("omega", "P", "Q")):
        axis.plot(time, trace, label=label)
        axis.axvline(STEP_TIME, color="black", linestyle=":", label="P_ref step")
        axis.set_ylabel(f"{label} (pu)")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("IEEE9 with one GFM: direct perturbation validation")
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(f"plot={OUTPUT}")


if __name__ == "__main__":
    main()
