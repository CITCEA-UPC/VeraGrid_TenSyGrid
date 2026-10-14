"""Exercise the experimental dynamic multilinear current limiter."""

from pathlib import Path
import sys

# Allow direct execution by file path as well as `python -m` from the repo.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np

from TensyGridEngine.emt.emt_explicit_converter_demo.utils.gfl_ode_time_inputs import (
    build_gfl_ode_problem,
    runtime_input_traces,
)
from TensyGridEngine.emt.emt_explicit_converter_demo.utils.simulation_data import case_dir, save_case_data
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Var, heaviside, sin

OUTPUT_DIR = case_dir("standalone_gfl_ode")
OUTPUT = OUTPUT_DIR / "comparison.png"


def main() -> None:
    time = Var("gfl_ode_time")
    phase_angles = (0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0)
    # This experiment has clean grid voltage. Only P_ref changes at 0.02 s;
    # the large command exercises the current limiter.
    inputs = {
        f"vg_{phase}": Const(np.sqrt(2.0))
        * sin(Const(2.0 * np.pi * 50.0) * time + Const(angle))
        for phase, angle in zip("ABC", phase_angles)
    }
    inputs.update({
        "P_dc_source": Const(0.5),
        "P_ref": Const(0.5) + Const(40.0 - 0.5) * heaviside(time - Const(0.02)),
        "Q_ref": Const(0.0),
    })
    problem = build_gfl_ode_problem(
        multilinear=True,
        dc_state="energy",
        enable_current_limiter=True,
        p0=0.5,
        q0=0.0,
        voltage=1.0 + 0.0j,
        prescribed_inputs=inputs,
        time_variable=time,
        simulation_time=0.06,
    )
    S, phi = problem.build_multilinear_matrices()
    result = problem.simulate()
    if not result.initialized or not result.converged:
        raise RuntimeError("Standalone GFL simulation failed")
    inputs = runtime_input_traces(problem, result.time)
    state_vars = problem.get_state_vars()
    input_names = [variable.name for variable in problem.sys_block.in_vars]
    save_case_data(
        OUTPUT_DIR, S=S, Phi=phi, time=result.time,
        states=np.column_stack([problem.trace(result, variable) for variable in state_vars]),
        state_names=[variable.name for variable in state_vars],
        inputs=np.column_stack([inputs[name] for name in input_names]),
        input_names=input_names,
        basis_names=[variable.name for variable in state_vars] + input_names,
    )

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
