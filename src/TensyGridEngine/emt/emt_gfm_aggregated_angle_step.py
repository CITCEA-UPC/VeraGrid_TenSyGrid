"""Open-loop angle-polarity test for the aggregated EMT GFM model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


HERE = Path(__file__).resolve().parent
VALIDATION_DIR = HERE.parents[3] / "VeraGrid" / "trunk" / "dynamics" / "model_validation"
OUTPUT_DIR = HERE / "results" / "gfm_aggregated_angle_step"

os.environ.setdefault("VERAGRID_GFM_EMT_SLACK_SOURCE", "balanced")
os.environ.setdefault("VERAGRID_GFM_EMT_TIME_STEP", "1e-5")
os.environ.setdefault("VERAGRID_GFM_EMT_COMPUTE_DENSE_COND", "0")
os.environ.setdefault("VERAGRID_EMT_INIT_CACHE_DIR", "/tmp/gfm-aggregated-angle-cache")

sys.path.insert(0, str(VALIDATION_DIR))

from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.enumerations import EmtInitializationMethod, EmtSolverTypes

import emt_gfm_model_validation as harness


STEP_RAD = float(os.environ.get("VERAGRID_GFM_ANGLE_STEP_RAD", "0.001"))
SIM_TIME = float(os.environ.get("VERAGRID_GFM_ANGLE_TEST_TIME", "0.001"))


def run_case(physical_angle_step: float) -> dict[str, np.ndarray]:
    grid, devices = harness.create_simplified_grid()
    pf = harness.run_power_flow(grid)
    if not pf.converged:
        raise RuntimeError("Power flow did not converge")

    gfm = harness.attach_emt_models(grid, devices, pf_results=pf)
    bus_idx = grid.buses.index(devices["bus_gfm"])
    voltage = complex(pf.voltage[bus_idx])
    p0 = float(np.real(pf.Sbus[bus_idx]) / grid.Sbase)
    q0 = float(np.imag(pf.Sbus[bus_idx]) / grid.Sbase)
    harness.set_block_event_value(gfm, grid.var_factory, "P_ref", p0)
    harness.set_block_event_value(gfm, grid.var_factory, "Q_ref", q0)
    harness.set_block_event_value(gfm, grid.var_factory, "V_ref", float(np.sqrt(2.0) * abs(voltage)))
    # Freeze the outer P-frequency feedback. The angle offset is then held
    # relative to the infinite bus because both sources rotate at 1 pu.
    harness.set_block_event_value(gfm, grid.var_factory, "Kdp", 0.0)

    options = EmtOptions(
        time_step=harness.TIME_STEP,
        simulation_time=SIM_TIME,
        tolerance=1e-6,
        integration_method=harness.integration_method_from_env(),
        solver_type=EmtSolverTypes.StructuralAD,
        initialization_method=EmtInitializationMethod.Explicit,
        init_newton_max_iter=200,
        verbose=0,
    )
    problem = EmtProblemDae(grid=grid, options=options, pf_results=pf)
    harness.seed_gfm_from_power_flow(problem, grid, pf, gfm, devices["bus_gfm"])
    harness.seed_bus_algebraic_predictors(problem, grid, pf)
    harness.seed_line_current_predictors(problem, grid, pf, devices)

    theta = harness.find_name_in_block("theta", gfm)
    if theta is None:
        raise RuntimeError("GFM theta state was not found")
    # With the conventional positive-angle Park pair, the imposed converter
    # frame displacement has the same sign as the theta-state displacement.
    problem.init_guess[theta.uid] += physical_angle_step

    time, values, _dy, well, converged = harness.run_emt(problem, options, SIM_TIME)
    if not well or not converged:
        raise RuntimeError(
            f"Angle case {physical_angle_step:+.3e} failed: well={well}, converged={converged}"
        )

    return {
        "time": time,
        "power": harness.trace(problem, values, "P", gfm),
        "vd_error": harness.trace(problem, values, "vd_ref", gfm) - harness.trace(problem, values, "vd_f", gfm),
        "iq_error": harness.trace(problem, values, "iq_ref", gfm) - harness.trace(problem, values, "iq_c", gfm),
    }


def main() -> None:
    cases = {
        "baseline": run_case(0.0),
        "+angle": run_case(STEP_RAD),
        "-angle": run_case(-STEP_RAD),
    }
    time = cases["baseline"]["time"]
    base_power = cases["baseline"]["power"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "angle_step_polarity.png"
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for label, case in cases.items():
        axes[0].plot(time * 1e3, case["power"], label=label)
        axes[1].plot(time * 1e3, case["power"] - base_power, label=label)
        axes[2].plot(time * 1e3, case["vd_error"], label=f"vd error, {label}")
    axes[0].set_ylabel("P (pu)")
    axes[1].set_ylabel("P - baseline (pu)")
    axes[2].set_ylabel("vd ref - vd (pu)")
    axes[2].set_xlabel("Time (ms)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle("Aggregated EMT GFM: frozen-droop angle-polarity test")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    probe_idx = min(len(time) - 1, max(1, int(round(0.0005 / harness.TIME_STEP))))
    dp_plus = float(cases["+angle"]["power"][probe_idx] - base_power[probe_idx])
    dp_minus = float(cases["-angle"]["power"][probe_idx] - base_power[probe_idx])
    slope = (dp_plus - dp_minus) / (2.0 * STEP_RAD)
    print(f"probe_time={time[probe_idx]:.9e}")
    print(f"delta_p_plus={dp_plus:.9e}")
    print(f"delta_p_minus={dp_minus:.9e}")
    print(f"central_dP_dphysical_angle={slope:.9e}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
