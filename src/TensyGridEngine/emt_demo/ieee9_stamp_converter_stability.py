#!/usr/bin/env python3
"""Test the public STAMP GFOR/GFOL EMT models on their IEEE9/WSCC case.

The STAMP models are linear deviation models around a published operating
point.  For that reason this demo deliberately uses their matching network,
dispatch, and one-SG/one-GFOR/one-GFOL source allocation.  Testing them on a
different dispatch would test an initialization mismatch, not the controls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np


STAMP_ROOT = Path("/home/pablo/Desktop/eroots/STAMP_Public")
STAMP_PERIODIC_X0 = STAMP_ROOT / "scripts/stamp_wscc_emt_periodic_x0.csv"
if str(STAMP_ROOT) not in sys.path:
    sys.path.insert(0, str(STAMP_ROOT))

from veragrid_stamp.emt_case import build_stamp_wscc_emt_grid


def _variable(model, suffix: str):
    variables = list(model.get_all_vars()) + list(model.event_dict)
    return next(var for var in variables if var.name.endswith(suffix))


def _trace(problem, values: np.ndarray, model, suffix: str) -> np.ndarray:
    variable = _variable(model, suffix)
    return values[:, problem.get_var_idx(variable)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-step", type=float, default=20e-6)
    parser.add_argument("--simulation-time", type=float, default=0.1)
    parser.add_argument("--no-event", action="store_true")
    parser.add_argument("--ssa", action="store_true", help="run EMT Floquet SSA after the time-domain check")
    parser.add_argument(
        "--periodic-oracle", action="store_true",
        help="diagnostic only: replace explicit x0 with STAMP's shooting solution",
    )
    args = parser.parse_args()

    from VeraGridEngine.Devices.Events.emt_event import EmtEvent
    from VeraGridEngine.Devices.Events.emt_events_group import EmtEventsGroup
    from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
    from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
    from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
    from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
    from VeraGridEngine.Simulations.PowerFlow.power_flow_options import PowerFlowOptions
    from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
    from VeraGridEngine.enumerations import (
        DynamicIntegrationMethod,
        EmtInitializationMethod,
        EmtSolverTypes,
    )

    grid = build_stamp_wscc_emt_grid()
    devices = {generator.name: generator for generator in grid.generators}
    if not args.no_event:
        # A short +1% system-base active-power request to the GFOL exercises
        # its PLL, P controller, inner current loop, and the GFOR/SG response.
        group = EmtEventsGroup(name="STAMP_GFOL_active_power_pulse")
        grid.add_emt_events_group(group)
        p_ref = _variable(devices["STAMP GFOL2"].emt_model, ".P_ref")
        grid.add_emt_event(EmtEvent(device=devices["STAMP GFOL2"], parameter=p_ref,
                                    time=0.02, value=0.01, group=group))
        grid.add_emt_event(EmtEvent(device=devices["STAMP GFOL2"], parameter=p_ref,
                                    time=0.04, value=0.0, group=group))

    pf_options = PowerFlowOptions(retry_with_other_methods=True)
    pf = PowerFlowDriver(grid=grid, options=pf_options)
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=pf_options)
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError("The matching STAMP IEEE9 power flow did not converge")

    options = EmtOptions(
        time_step=args.time_step,
        simulation_time=args.simulation_time,
        tolerance=1e-8,
        solver_type=EmtSolverTypes.StructuralCompiled,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        # STAMP's q-d/abc interface and its sqrt(3) inverse-Park factor are
        # formulated on VeraGrid's conventional total-three-phase power base.
        conventional_three_phase_base=True,
        verbose=0,
    )
    problem = EmtProblemDae(grid=grid, options=options,
                            pf_results=pf.results, pf_results_3ph=pf3.results)
    periodic_x0 = np.loadtxt(STAMP_PERIODIC_X0, delimiter=",")
    all_variables = problem.get_state_vars() + problem.get_algebraic_vars()
    if periodic_x0.shape != (len(all_variables),):
        raise ValueError(
            f"STAMP periodic x0 has shape {periodic_x0.shape}; "
            f"this model requires {(len(all_variables),)}"
        )
    explicit_x0 = problem.get_x0().copy()
    delta = periodic_x0 - explicit_x0
    print("Largest explicit-x0 errors relative to periodic shooting:")
    for index in np.argsort(np.abs(delta))[::-1][:20]:
        print(f"  delta={delta[index]:+.6e} explicit={explicit_x0[index]:+.6e} "
              f"oracle={periodic_x0[index]:+.6e}  {all_variables[index].name}")
    if args.periodic_oracle:
        # Diagnostic oracle only. The normal test must prove that symbolic
        # explicit initialization can produce the same periodic orbit.
        for variable, value in zip(all_variables, periodic_x0):
            problem.init_guess[variable.uid] = float(value)
        from VeraGridEngine.Simulations.EMT.initialization_emt import _compute_missing_dx0
        static_params = np.asarray(
            [float(parameter.value) for parameter in problem.get_parameters_values()]
        )
        _compute_missing_dx0(
            problem=problem,
            report=problem.initialization_report,
            x_full=problem.get_x0(),
            dx_full=problem.get_dx0(),
            runtime_params=problem.event_params_values.copy(),
            constant_params=static_params,
            include_existing=True,
        )
    if not args.no_event:
        problem.set_events_group(grid.emt_events_groups[-1])
    solver = build_emt_solver(options=options, problem=problem, t0=0.0,
                              t_end=args.simulation_time, h=args.time_step,
                              method=options.integration_method)
    time, values, derivatives, initialized, converged = solver.simulate(
        boundary_updater=cast(Any, problem))

    sg = devices["STAMP SG1"].emt_model
    gfor = devices["STAMP GFOR1"].emt_model
    gfol = devices["STAMP GFOL2"].emt_model
    signals = {
        "SG speed": _trace(problem, values, sg, ".w_pu") - 1.0,
        "GFOR angle": _trace(problem, values, gfor, ".etheta_x"),
        "GFOL angle": _trace(problem, values, gfol, ".etheta_x"),
        "GFOR $P_f$": _trace(problem, values, gfor, ".p_filt_x"),
        "GFOR $Q_f$": _trace(problem, values, gfor, ".q_filt_x"),
        "GFOR $i_{gq}$": _trace(problem, values, gfor, ".ig_q"),
        "GFOL $i_{gq}$": _trace(problem, values, gfol, ".ig_q"),
    }

    report = problem.initialization_report
    print(f"initialized={initialized}, converged={converged}, finite={np.isfinite(values).all()}")
    if report is not None:
        print(f"initialization={report.status.name}, residual={report.final_residual_inf:.3e}")
    for name, trace in signals.items():
        settled = trace[time >= min(0.02, 0.5*args.simulation_time)]
        print(f"{name:14s}: final={trace[-1]:+.6e}, "
              f"peak={np.max(np.abs(trace)):.6e}, "
              f"post-init peak={np.max(np.abs(settled)):.6e}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                             constrained_layout=True)
    axes[0].plot(1e3*time, signals["SG speed"], label="SG speed deviation")
    axes[0].plot(1e3*time, signals["GFOR angle"], label="GFOR angle deviation")
    axes[0].plot(1e3*time, signals["GFOL angle"], label="GFOL PLL angle deviation")
    axes[1].plot(1e3*time, signals["GFOR $P_f$"], label="GFOR filtered P deviation")
    axes[1].plot(1e3*time, signals["GFOR $Q_f$"], label="GFOR filtered Q state")
    axes[2].plot(1e3*time, signals["GFOR $i_{gq}$"], label="GFOR grid-current deviation")
    axes[2].plot(1e3*time, signals["GFOL $i_{gq}$"], label="GFOL grid-current deviation")
    axes[0].set_ylabel("speed / angle")
    axes[1].set_ylabel("power state (pu)")
    axes[2].set_ylabel("current (pu)")
    axes[2].set_xlabel("time (ms)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        if not args.no_event:
            axis.axvspan(20.0, 40.0, color="tab:orange", alpha=0.12)
    axes[0].set_title("STAMP converters on their matching IEEE9/WSCC operating point")
    suffix = "no_event" if args.no_event else "p_ref_pulse"
    output = Path(__file__).with_name(f"ieee9_stamp_converter_stability_{suffix}.png")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    np.savez(output.with_suffix(".npz"), time=time, **signals)
    print(f"plot={output}")

    if args.ssa:
        from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_driver import (
            SmallSignalStabilityEmtDriver,
        )
        from VeraGridEngine.Simulations.SmallSignalStabilityEmt.small_signal_stability_emt_options import (
            SmallSignalStabilityEmtOptions,
        )

        ssa_options = SmallSignalStabilityEmtOptions(
            k=min(30, problem.get_states_number() - 2),
            target_period=1.0 / grid.fBase,
            ss_assessment_time=0.06,
            verbose=1,
        )
        driver = SmallSignalStabilityEmtDriver(
            grid=grid,
            emt_options=options,
            sss_options=ssa_options,
            pf_results=pf3.results,
        )
        driver.run()
        eigenvalues = np.asarray(driver.results.eigenvalues, dtype=complex)
        order = np.argsort(eigenvalues.real)[::-1]
        print("Rightmost Floquet exponents:")
        for index in order[: min(15, len(order))]:
            value = eigenvalues[index]
            multiplier_magnitude = np.exp(value.real / grid.fBase)
            print(
                f"  lambda={value.real:+.8e}{value.imag:+.8e}j "
                f"|mu|={multiplier_magnitude:.10f}"
            )

        figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        axis.scatter(eigenvalues.real, eigenvalues.imag / (2.0 * np.pi), s=22)
        axis.axvline(0.0, color="black", linewidth=1.0)
        axis.grid(True, alpha=0.3)
        axis.set_xlabel(r"$\mathrm{Re}(\lambda)$ (1/s)")
        axis.set_ylabel(r"$\mathrm{Im}(\lambda)/(2\pi)$ (Hz)")
        axis.set_title("STAMP IEEE9 EMT Floquet exponents")
        eigen_plot = Path(__file__).with_name("ieee9_stamp_converter_floquet.png")
        figure.savefig(eigen_plot, dpi=180)
        plt.close(figure)
        print(f"ssa_plot={eigen_plot}")


if __name__ == "__main__":
    main()
