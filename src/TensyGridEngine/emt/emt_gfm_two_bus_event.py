"""Two-bus EMT event validation for the detailed UPC grid-forming VSC.

The test uses the production ``emt_gfm_upc`` template, initializes it from an
AC power flow, applies a small active-power-reference step, and plots both the
outer response and the inner voltage/current-control errors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


HERE = Path(__file__).resolve().parent
TENSYGRID_ROOT = HERE.parents[2]
EROOTS_ROOT = TENSYGRID_ROOT.parent
VALIDATION_DIR = EROOTS_ROOT / "VeraGrid" / "trunk" / "dynamics" / "model_validation"
OUTPUT_DIR = HERE / "results" / "gfm_two_bus_event"

# Set test defaults before importing the validation harness, which reads its
# configuration at module-import time. Every value remains user-overridable.
os.environ.setdefault("VERAGRID_GFM_EMT_SLACK_SOURCE", "balanced")
os.environ.setdefault("VERAGRID_GFM_EMT_TIME_STEP", "1e-5")
os.environ.setdefault("VERAGRID_GFM_EMT_SIM_TIME", "0.20")
os.environ.setdefault("VERAGRID_GFM_EMT_P_REF_STEP_TIME", "0.02")
os.environ.setdefault("VERAGRID_GFM_EMT_P_REF_STEP", "0.01")
os.environ.setdefault("VERAGRID_GFM_EMT_COMPUTE_DENSE_COND", "0")

sys.path.insert(0, str(VALIDATION_DIR))

import VeraGridEngine.api as vge
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Templates.Emt.emt_gfm_upc import build_emt_gfm_aggregated_model
from VeraGridEngine.enumerations import EmtInitializationMethod, EmtSolverTypes

import emt_gfm_model_validation as harness


def _tail_span(values: np.ndarray) -> float:
    tail = values[max(0, int(0.9 * values.size)):]
    return float(np.ptp(tail)) if tail.size else float("nan")


def run() -> tuple[Path, dict[str, float]]:
    grid, devices = harness.create_simplified_grid()
    pf_results = harness.run_power_flow(grid)
    if not pf_results.converged:
        raise RuntimeError("The two-bus AC power flow did not converge")

    # This assertion makes the test fail if the harness ever falls back to a
    # validation-only copy instead of the installed EMT template.
    if harness.build_emt_gfm_model.__globals__["build_emt_gfm_aggregated_model"] is not build_emt_gfm_aggregated_model:
        raise RuntimeError("Two-bus test is not using the VeraGrid EMT template")

    gfm = harness.attach_emt_models(grid, devices, pf_results=pf_results)
    bus_idx = grid.buses.index(devices["bus_gfm"])
    voltage = complex(pf_results.voltage[bus_idx])
    p0 = float(np.real(pf_results.Sbus[bus_idx]) / grid.Sbase)
    q0 = float(np.imag(pf_results.Sbus[bus_idx]) / grid.Sbase)
    harness.set_block_event_value(gfm, grid.var_factory, "P_ref", p0)
    harness.set_block_event_value(gfm, grid.var_factory, "Q_ref", q0)
    harness.set_block_event_value(gfm, grid.var_factory, "V_ref", float(np.sqrt(2.0) * abs(voltage)))

    options = EmtOptions(
        time_step=harness.TIME_STEP,
        simulation_time=harness.SIM_TIME,
        tolerance=1.0e-6,
        integration_method=harness.integration_method_from_env(),
        solver_type=EmtSolverTypes.StructuralAD,
        initialization_method=EmtInitializationMethod.Explicit,
        init_newton_max_iter=200,
        verbose=0,
    )
    problem = EmtProblemDae(grid=grid, options=options, pf_results=pf_results)
    harness.seed_gfm_from_power_flow(problem, grid, pf_results, gfm, devices["bus_gfm"])
    harness.seed_bus_algebraic_predictors(problem, grid, pf_results)
    harness.seed_line_current_predictors(problem, grid, pf_results, devices)

    step_time = harness.P_REF_STEP_TIME
    step_delta = float(harness.P_REF_STEP or 0.01)
    time, values, _derivatives, well_initialized, converged = harness.run_emt_with_p_ref_step(
        problem, options, gfm, step_time=step_time, step_delta=step_delta
    )
    if not well_initialized or not converged:
        raise RuntimeError(
            f"EMT solve failed: well_initialized={well_initialized}, converged={converged}"
        )

    signal = lambda name: harness.trace(problem, values, name, gfm)
    omega = signal("omega")
    power = signal("P")
    power_lp = signal("y_p_lp")
    reactive = signal("Q")
    voltage = signal("V")
    vd_error = signal("vd_ref") - signal("vd_f")
    vq_error = signal("vq_ref") - signal("vq_f")
    id_error = signal("id_ref") - signal("id_c")
    iq_error = signal("iq_ref") - signal("iq_c")
    theta_gfm = signal("theta")
    vd_c = signal("vd_c")
    vq_c = signal("vq_c")
    theta_source = harness.trace(problem, values, "theta_src", devices["gen_slack"].emt_model)
    converter_phase = np.unwrap(np.pi - theta_gfm - np.arctan2(vd_c, vq_c))
    # The source phase offset is constant, so it cancels from angle changes.
    source_phase = np.unwrap(theta_source)
    relative_phase = np.unwrap(converter_phase - source_phase)
    p_ref = np.where(time < step_time, p0, p0 + step_delta)

    pre = time < step_time
    post = time >= step_time
    metrics = {
        "pre_omega_span": float(np.ptp(omega[pre])),
        "pre_power_span": float(np.ptp(power_lp[pre])),
        "peak_frequency_deviation_hz": float(np.max(np.abs(omega[post] - 1.0)) * grid.fBase),
        "final_power_error": float(power_lp[-1] - p_ref[-1]),
        "tail_omega_span": _tail_span(omega),
        "tail_power_span": _tail_span(power_lp),
        "tail_voltage_error_span": max(_tail_span(vd_error), _tail_span(vq_error)),
        "tail_current_error_span": max(_tail_span(id_error), _tail_span(iq_error)),
        "relative_phase_change": float(relative_phase[-1] - relative_phase[np.flatnonzero(post)[0]]),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "gfm_two_bus_event.png"
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(time, power, label="P")
    axes[0].plot(time, power_lp, label="P filtered")
    axes[0].plot(time, p_ref, "--", label="P reference")
    axes[0].set_ylabel("Active power (pu)")
    axes[0].legend()
    axes[1].plot(time, (omega - 1.0) * grid.fBase, label="Frequency deviation")
    axes[1].plot(time, relative_phase - relative_phase[0], label="Converter-grid angle change (rad)")
    axes[1].set_ylabel("Frequency deviation (Hz)")
    axes[1].legend()
    axes[2].plot(time, reactive, label="Q")
    axes[2].plot(time, voltage, label="Voltage reference output")
    axes[2].set_ylabel("Q / voltage (pu)")
    axes[2].legend()
    axes[3].plot(time, vd_error, label="vd ref - vd")
    axes[3].plot(time, vq_error, label="vq ref - vq")
    axes[3].plot(time, id_error, label="id ref - id")
    axes[3].plot(time, iq_error, label="iq ref - iq")
    axes[3].set_ylabel("Inner-loop error (pu)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(ncol=2)
    for axis in axes:
        axis.axvline(step_time, color="0.35", linestyle=":", linewidth=1.0)
        axis.grid(True, alpha=0.3)
    fig.suptitle("Two-bus UPC EMT GFM: active-power reference event")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"template={build_emt_gfm_aggregated_model.__module__}")
    print(f"well_initialized={well_initialized} converged={converged} samples={time.size}")
    for name, value in metrics.items():
        print(f"{name}={value:.9e}")
    print(f"plot={plot_path}")
    return plot_path, metrics


if __name__ == "__main__":
    run()
