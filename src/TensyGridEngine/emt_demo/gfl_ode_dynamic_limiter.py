"""Exercise the experimental dynamic multilinear current limiter."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt_demo.gfl_ode_time_inputs import (
    build_gfl_ode_problem,
    runtime_input_traces,
)

OUTPUT = Path(__file__).with_name("gfl_ode_dynamic_limiter.png")


def main() -> None:
    problem = build_gfl_ode_problem(
        multilinear=True,
        dc_state="energy",
        enable_current_limiter=True,
        p0=0.5,
        q0=0.0,
        voltage=1.0 + 0.0j,
        voltage_noise_rms=0.0,
        # Deliberately extreme command: the slow outer-loop gains otherwise do
        # not reach the 2.5 pu current limit within this short EMT experiment.
        p_reference_step=(0.02, 40.0),
        simulation_time=0.06,
    )
    S, phi = problem.build_multilinear_matrices()
    result = problem.simulate()
    inputs = runtime_input_traces(problem, result.time)

    iq_limited = problem.trace(result, "i_q_ref_sat_out")
    id_limited = problem.trace(result, "i_d_ref_sat_out")
    current_magnitude = np.sqrt(iq_limited ** 2 + id_limited ** 2)

    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(result.time, inputs["P_ref"], label="P reference")
    axes[0].set_ylabel("P reference (pu)")
    axes[1].plot(result.time, iq_limited, label="limited q-axis current")
    axes[1].axhline(2.5, color="black", linestyle="--", label="Imax")
    axes[1].set_ylabel("Current (pu peak)")
    axes[1].legend()
    axes[2].plot(result.time, current_magnitude, label="limited dq magnitude")
    axes[2].axhline(2.5, color="black", linestyle="--", label="Imax")
    axes[2].set_ylabel("Magnitude (pu peak)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.suptitle("Dynamic multilinear current-limiter experiment")
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)

    print(f"initialized={result.initialized}, converged={result.converged}")
    print(f"S={S.shape}, Phi={phi.shape}")
    print(f"peak_limited_current={np.max(current_magnitude):.9f} pu")
    print(f"plot={OUTPUT}")


if __name__ == "__main__":
    main()
