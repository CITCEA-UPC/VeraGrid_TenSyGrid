"""Check whether the explicit GFL ODE follows the older block-based GFL.

Both models start at the same operating point, see the same clean 50 Hz grid
voltage and DC-source current, and receive the same P_ref and Q_ref steps.
The script compares their shared states and measured P/Q trajectories. Large
opposite responses would be evidence of a sign or convention mismatch.

This is a numerical comparison, not a symbolic proof that every equation is
identical. It uses the voltage-state ODE because the standalone original model
also has a dynamic DC-voltage state; the IEEE9 stiff-DC variant is different.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from TensyGridEngine.emt.emt_explicit_converter_demo.utils.gfl_ode_time_inputs import build_gfl_ode_problem
from TensyGridEngine.emt.emt_explicit_converter_demo.experiments.single_model_time_inputs import build_problem
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Var, heaviside, sin
from VeraGridEngine.enumerations import DynamicIntegrationMethod

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "simulation_data" / "experiments" / "gfl_ode_model_comparison"
P0 = 0.25                 # Initial active power (pu).
Q0 = 0.02                 # Initial reactive power (pu).
VDC0 = 2.0                # Initial DC voltage (pu).
P_STEP_TIME = 0.01        # Seconds.
Q_STEP_TIME = 0.03        # Seconds.
P_STEP_SIZE = 0.05        # pu; positive to expose active-power sign.
Q_STEP_SIZE = 0.02        # pu; positive to expose reactive-power sign.
SIMULATION_TIME = 0.05   # Seconds.
TIME_STEP = 5.0e-5        # Same numerical step for both models.

# These names exist as states in both models. Comparing each trace checks more
# than just the terminal power response.
SHARED_STATES = (
    "i_line_A", "i_line_B", "i_line_C", "Vdc_", "theta", "xi_PLL",
    "xi_Pac_ctrl", "xi_Qac_ctrl", "xi_vd_hat", "xi_vq_hat",
)


def reference_expressions(time: Var):
    """The same commanded power changes are applied to both controllers."""
    p_ref = Const(P0) + Const(P_STEP_SIZE) * heaviside(time - Const(P_STEP_TIME))
    q_ref = Const(Q0) + Const(Q_STEP_SIZE) * heaviside(time - Const(Q_STEP_TIME))
    return p_ref, q_ref


def build_matching_problems():
    """Build the original DAE and explicit ODE with matching test conditions."""
    original = build_problem(
        "gfl", multilinear=False, p0=P0, q0=Q0,
        voltage=1.0 + 0.0j, vdc0=VDC0,
        voltage_noise_rms=0.0, dc_side="power_balance",
        simulation_time=SIMULATION_TIME,
    )
    original_p_ref, original_q_ref = reference_expressions(original.glob_time)
    original.set_runtime_expression("P_ref", original_p_ref)
    original.set_runtime_expression("Q_ref", original_q_ref)

    time = Var("gfl_ode_time")
    angle = Const(2.0 * np.pi * 50.0) * time
    phase_shift = Const(2.0 * np.pi / 3.0)
    p_ref, q_ref = reference_expressions(time)
    # The ODE builder receives the full external input vector explicitly.
    ode_inputs = {
        "vg_A": Const(np.sqrt(2.0)) * sin(angle),
        "vg_B": Const(np.sqrt(2.0)) * sin(angle - phase_shift),
        "vg_C": Const(np.sqrt(2.0)) * sin(angle + phase_shift),
        "i_dc_source": Const(P0 / VDC0),
        "P_ref": p_ref,
        "Q_ref": q_ref,
    }
    ode = build_gfl_ode_problem(
        multilinear=False, dc_state="voltage", p0=P0, q0=Q0,
        voltage=1.0 + 0.0j, vdc0=VDC0,
        prescribed_inputs=ode_inputs, time_variable=time,
        simulation_time=SIMULATION_TIME,
    )

    # The existing standalone original uses backward Euler. Match the ODE's
    # integration method and step size so solver choice is not the main source
    # of differences.
    original.options.time_step = TIME_STEP
    ode.options.time_step = TIME_STEP
    ode.options.integration_method = DynamicIntegrationMethod.DaeBackEuler
    return original, ode


def ode_power_traces(ode, result):
    """Calculate the ODE's P and Q using the same Park convention as its model."""
    time = result.time
    angle = ode.trace(result, "theta")
    cosine = np.cos(angle)
    sine = np.sin(angle)
    vg_a = np.sqrt(2.0) * np.sin(2.0 * np.pi * 50.0 * time)
    vg_b = np.sqrt(2.0) * np.sin(2.0 * np.pi * 50.0 * time - 2.0 * np.pi / 3.0)
    vg_c = np.sqrt(2.0) * np.sin(2.0 * np.pi * 50.0 * time + 2.0 * np.pi / 3.0)
    ia = ode.trace(result, "i_line_A")
    ib = ode.trace(result, "i_line_B")
    ic = ode.trace(result, "i_line_C")
    sqrt3 = np.sqrt(3.0)
    vg_d = (2.0 * cosine * vg_a + (-cosine + sqrt3 * sine) * vg_b
            + (-cosine - sqrt3 * sine) * vg_c) / 3.0
    vg_q = (2.0 * sine * vg_a + (-sine - sqrt3 * cosine) * vg_b
            + (-sine + sqrt3 * cosine) * vg_c) / 3.0
    i_d = (2.0 * cosine * ia + (-cosine + sqrt3 * sine) * ib
           + (-cosine - sqrt3 * sine) * ic) / 3.0
    i_q = (2.0 * sine * ia + (-sine - sqrt3 * cosine) * ib
           + (-sine + sqrt3 * cosine) * ic) / 3.0
    p = 0.5 * (vg_q * i_q + vg_d * i_d)
    q = 0.5 * (vg_d * i_q - vg_q * i_d)
    return p, q


def main() -> None:
    original, ode = build_matching_problems()
    original_result = original.simulate()
    ode_result = ode.simulate()
    if not (original_result.initialized and original_result.converged
            and ode_result.initialized and ode_result.converged):
        raise RuntimeError("Both models must initialize and converge for comparison")
    if not np.array_equal(original_result.time, ode_result.time):
        raise RuntimeError("The models returned different sample times")

    time = original_result.time
    labels = (*SHARED_STATES, "P", "Q")
    original_traces = [original.trace(original_result, name) for name in SHARED_STATES]
    ode_traces = [ode.trace(ode_result, name) for name in SHARED_STATES]
    original_traces.extend((original.trace(original_result, "P"),
                            original.trace(original_result, "Q")))
    ode_traces.extend(ode_power_traces(ode, ode_result))
    original_values = np.column_stack(original_traces)
    ode_values = np.column_stack(ode_traces)
    difference = ode_values - original_values

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    savemat(OUTPUT_DIR / "comparison.mat", {
        "time": time.reshape(-1, 1),
        "signal_names": np.asarray(labels, dtype=object).reshape(1, -1),
        "original": original_values,
        "explicit_ode": ode_values,
        "difference_ode_minus_original": difference,
    }, do_compression=True)

    # Plot the quantities whose direction would change with a power-control
    # sign error, plus one physical current state for a waveform comparison.
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for axis, name in ((axes[0], "P"), (axes[1], "Q"), (axes[2], "i_line_A")):
        index = labels.index(name)
        axis.plot(time, original_values[:, index], label="original block-based GFL")
        axis.plot(time, ode_values[:, index], linestyle="--", label="explicit GFL ODE")
        axis.set_ylabel(f"{name} (pu)")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].axvline(P_STEP_TIME, color="black", alpha=0.3)
    axes[1].axvline(Q_STEP_TIME, color="black", alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    figure.savefig(OUTPUT_DIR / "comparison.png", dpi=180)
    plt.close(figure)

    print("ODE minus original, with identical operating point and input steps:")
    for index, name in enumerate(labels):
        error = difference[:, index]
        print(f"{name:>14}: max_abs={np.max(np.abs(error)):.6g} pu, "
              f"rms={np.sqrt(np.mean(error ** 2)):.6g} pu")
    for name, step_time in (("P", P_STEP_TIME), ("Q", Q_STEP_TIME)):
        index = labels.index(name)
        before = np.searchsorted(time, step_time) - 1
        original_change = original_values[-1, index] - original_values[before, index]
        ode_change = ode_values[-1, index] - ode_values[before, index]
        print(f"{name} step response: original={original_change:+.6g} pu, "
              f"ODE={ode_change:+.6g} pu")
    print(f"results={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
