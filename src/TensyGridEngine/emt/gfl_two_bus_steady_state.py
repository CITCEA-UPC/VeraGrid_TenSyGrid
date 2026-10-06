"""Two-bus EMT regression for a GFL model attached directly as a generator."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.initialization_emt import run_emt_native_initialization
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Devices.Events.emt_event import EmtEvent
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.balanced_source_emt_template import (
    get_balanced_3ph_voltage_source_emt_template,
)
from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import (
    get_generator_thevenin_rl_emt_template_with_ref,
)
from VeraGridEngine.Templates.Emt.vsc_gfl_emt import (
    VscGflEmtBuild,
    install_gfl_generator_initialization,
)
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import (
    ConverterControlType, DynamicIntegrationMethod, EmtInitializationMethod, EmtSolverTypes,
)

from TensyGridEngine.emt import ieee9_emt_ibr_simulation as ibr
from TensyGridEngine.emt.emt_ieee9 import build_emt_options, build_power_flow_options


TIME_STEP = float(os.environ.get("VERAGRID_GFL_2BUS_TIME_STEP", "5e-6"))
SIMULATION_TIME = float(os.environ.get("VERAGRID_GFL_2BUS_SIM_TIME", "0.04"))
SOLVER_TOLERANCE = float(os.environ.get("VERAGRID_GFL_2BUS_TOLERANCE", "1e-9"))
INTEGRATION_METHOD = DynamicIntegrationMethod[
    os.environ.get("VERAGRID_GFL_2BUS_INTEGRATION_METHOD", "DaeTrapezoidal")
]
P_MW = float(os.environ.get("VERAGRID_GFL_2BUS_P_MW", "80.0"))
Q_MVAR = float(os.environ.get("VERAGRID_GFL_2BUS_Q_MVAR", "0.0"))
SLACK_SOURCE = os.environ.get("VERAGRID_GFL_2BUS_SLACK_SOURCE", "balanced").lower()
P_EVENT_TIME = float(os.environ.get("VERAGRID_GFL_2BUS_P_EVENT_TIME", "-1.0"))
P_INJECTION_STEP = float(os.environ.get("VERAGRID_GFL_2BUS_P_INJECTION_STEP", "0.01"))
KP_POWER = float(os.environ.get("VERAGRID_GFL_2BUS_KP_POWER", "0.5"))
KI_POWER = float(os.environ.get("VERAGRID_GFL_2BUS_KI_POWER", "10.0"))
KP_CURRENT = float(os.environ.get("VERAGRID_GFL_2BUS_KP_CURRENT", "1.0"))
KI_CURRENT = float(os.environ.get("VERAGRID_GFL_2BUS_KI_CURRENT", "20.0"))
KP_PLL = float(os.environ.get("VERAGRID_GFL_2BUS_KP_PLL", "0.03"))
KI_PLL = float(os.environ.get("VERAGRID_GFL_2BUS_KI_PLL", "0.2"))
PLOT_OUTPUT_DIR = Path(os.environ.get(
    "VERAGRID_GFL_2BUS_PLOT_OUTPUT_DIR", str(Path(__file__).parent)
))


def build_grid():
    grid = gce.MultiCircuit(name="GFL two-bus steady-state test", Sbase=100.0, fbase=50.0)
    gfl_bus = gce.Bus(name="GFL bus", Vnom=230.0)
    infinite_bus = gce.Bus(name="Infinite bus", Vnom=230.0, is_slack=True)
    grid.add_bus(gfl_bus)
    grid.add_bus(infinite_bus)
    get_bus_emt_template(grid, gfl_bus)
    get_bus_emt_template(grid, infinite_bus)

    line = gce.Line(
        name="GFL-to-grid", bus_from=gfl_bus, bus_to=infinite_bus,
        r=0.01, x=0.10, b=0.0, rate=200.0,
    )
    gfl_generator = gce.Generator(
        name="GFL generator", P=P_MW, vset=1.0, Snom=100.0,
        r1=0.0, x1=0.0,
    )
    slack_generator = gce.Generator(
        name="Infinite source", P=0.0, vset=1.0, Snom=100.0,
        r1=0.001, x1=0.20,
    )
    grid.add_line(line)
    grid.add_generator(bus=gfl_bus, api_obj=gfl_generator)
    grid.add_generator(bus=infinite_bus, api_obj=slack_generator)
    return grid, line, gfl_generator, slack_generator


def build_models(grid, line, gfl_generator, slack_generator, pf_results):
    helper, _gfm_models, _gfm_validation = ibr._load_deliverable_models()
    line_model = get_pi_line_emt_template(grid.var_factory, name="two_bus_line").block
    slack_index = grid.buses.index(slack_generator.bus)
    voltage = complex(pf_results.voltage[slack_index])
    vpk = float(np.sqrt(2.0) * abs(voltage))
    angle = float(np.angle(voltage))
    p0 = float(np.real(pf_results.Sbus[slack_index]) / grid.Sbase)
    q0 = float(np.imag(pf_results.Sbus[slack_index]) / grid.Sbase)
    va0 = vpk * np.sin(angle)
    vb0 = vpk * np.sin(angle - 2.0 * np.pi / 3.0)
    vc0 = vpk * np.sin(angle + 2.0 * np.pi / 3.0)
    current_matrix = np.asarray(
        [
            [va0 / 3.0, vb0 / 3.0, vc0 / 3.0],
            [(vb0 - vc0) / (3.0 * np.sqrt(3.0)),
             (vc0 - va0) / (3.0 * np.sqrt(3.0)),
             (va0 - vb0) / (3.0 * np.sqrt(3.0))],
            [1.0, 1.0, 1.0],
        ]
    )
    ia0, ib0, ic0 = np.linalg.solve(current_matrix, np.asarray([p0, q0, 0.0]))
    source_g = 1000.0
    omega_base = 2.0 * np.pi * grid.fBase
    dva0 = omega_base * vpk * np.cos(angle)
    # A balanced positive-sequence current has the same phase derivative rule.
    dia0 = omega_base * (ic0 - ib0) / np.sqrt(3.0)
    vsrc_a0 = va0 + ia0 / source_g
    dvsrc_a0 = dva0 + dia0 / source_g
    source_amp = float(np.hypot(vsrc_a0, dvsrc_a0 / omega_base))
    source_angle_deg = float(np.rad2deg(np.arctan2(vsrc_a0, dvsrc_a0 / omega_base)))
    if SLACK_SOURCE == "thevenin":
        slack_model = get_generator_thevenin_rl_emt_template_with_ref(
            grid.var_factory, name="two_bus_thevenin_source"
        ).block
    else:
        slack_model = get_balanced_3ph_voltage_source_emt_template(
            grid.var_factory,
            amplitude_value=source_amp,
            frequency_hz=grid.fBase,
            phase_a_deg=source_angle_deg,
            source_conductance_value=source_g,
            name="two_bus_infinite_source",
        ).block
    gfl_model = VscGflEmtBuild(
        grid.var_factory,
        name="two_bus_gfl",
        control1=ConverterControlType.Pac,
        control2=ConverterControlType.Qac,
        frozen_voltage_source=False,
    ).block
    helper.set_event_value_in_block(grid.var_factory, "I_max", gfl_model, 2.0)
    helper.add_internal_filter_block(grid, gfl_model)
    helper.set_event_value_in_block(grid.var_factory, "R_filter", gfl_model, 0.01)
    helper.set_event_value_in_block(grid.var_factory, "Kp_pol", gfl_model, KP_POWER)
    helper.set_event_value_in_block(grid.var_factory, "Ki_pol", gfl_model, KI_POWER)
    helper.set_event_value_in_block(grid.var_factory, "Kp_icl", gfl_model, KP_CURRENT)
    helper.set_event_value_in_block(grid.var_factory, "Ki_icl", gfl_model, KI_CURRENT)
    helper.set_event_value_in_block(grid.var_factory, "Kp_pll", gfl_model, KP_PLL)
    helper.set_event_value_in_block(grid.var_factory, "Ki_pll", gfl_model, KI_PLL)
    ibr._correct_gfl_filter_rotation_convention(gfl_model)
    helper.connect_internal_filter_ports(grid, gfl_model)
    ibr._adapt_gfl_as_generator(gfl_model, grid.var_factory)

    set_emt_model(line, line_model, grid.var_factory)
    set_emt_model(slack_generator, slack_model, grid.var_factory)
    set_emt_model(gfl_generator, gfl_model, grid.var_factory)
    install_gfl_generator_initialization(gfl_generator.emt_model, grid.var_factory, grid.fBase)
    return helper, gfl_generator.emt_model


def trace(problem, values, block, name):
    signal = ibr._get_signal(problem, values, block, name)
    if signal is None:
        raise KeyError(name)
    return np.asarray(signal, dtype=float)


def cycle_mean_drift(time, signal, frequency=50.0):
    samples_per_cycle = max(1, int(round((1.0 / frequency) / TIME_STEP)))
    if signal.size < 2 * samples_per_cycle:
        return float(signal[-1] - signal[0])
    first = float(np.mean(signal[:samples_per_cycle]))
    last = float(np.mean(signal[-samples_per_cycle:]))
    return last - first


def main() -> None:
    grid, line, gfl_generator, slack_generator = build_grid()
    pf = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    pf.run()
    pf3 = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    pf3.run()
    if not bool(pf.results.converged) or not bool(pf3.results.converged):
        raise RuntimeError("Two-bus power flow did not converge")

    helper, gfl_model = build_models(grid, line, gfl_generator, slack_generator, pf.results)
    p_ref = ibr._find_combined_parameter(gfl_model, "P_ref")
    # Native VSC power is positive into the converter; the direct-generator
    # adapter exposes the opposite sign at the AC bus.
    p_ref_initial = -P_MW / grid.Sbase
    event_group = None
    if P_EVENT_TIME >= 0.0:
        # The VSC branch convention is positive into the converter, whereas this
        # device is exposed as a generator.  More generator injection therefore
        # means a more-negative native converter P reference.
        p_ref_event = p_ref_initial - P_INJECTION_STEP
        event_group = gce.EmtEventsGroup(name="GFL active-power reference event")
        grid.add_emt_events_group(event_group)
        grid.add_emt_event(EmtEvent(
            device=gfl_generator, parameter=p_ref,
            time=P_EVENT_TIME, value=p_ref_event, group=event_group,
        ))
        print(
            f"P-injection event: t={P_EVENT_TIME:g}s, "
            f"generator step={P_INJECTION_STEP:+.6g} pu, "
            f"native P_ref={p_ref_initial:.9g}->{p_ref_event:.9g}"
        )
    options = build_emt_options()
    options.solver_type = EmtSolverTypes.StructuralAD
    options.time_step = TIME_STEP
    options.simulation_time = SIMULATION_TIME
    options.tolerance = SOLVER_TOLERANCE
    options.integration_method = INTEGRATION_METHOD
    options.verbose = int(os.environ.get("VERAGRID_GFL_2BUS_VERBOSE", "0"))
    problem = EmtProblemDae(
        grid=grid, options=options, pf_results=pf.results, pf_results_3ph=pf3.results
    )
    if event_group is not None:
        problem.set_events_group(event_group)
    report = None
    if os.environ.get("VERAGRID_GFL_2BUS_POST_INIT", "0") != "0":
        old_method, old_tol = options.initialization_method, options.init_newton_tol
        options.initialization_method = EmtInitializationMethod.ConsistentNewton
        options.init_newton_tol = options.tolerance
        report = run_emt_native_initialization(problem=problem, options=options)
        options.initialization_method, options.init_newton_tol = old_method, old_tol
    x0_seed = problem.get_x0().copy()
    dx0_seed = problem.get_dx0().copy()

    solver = build_emt_solver(
        options=options, problem=problem, t0=0.0, t_end=SIMULATION_TIME,
        h=TIME_STEP, method=options.integration_method,
    )
    time, values, _derivatives, initialized, converged = solver.simulate(boundary_updater=problem)
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)

    signals = {
        "omega": trace(problem, values, gfl_model, "omega"),
        # Report bus injection for a generator-facing test; native VSC P/Q are
        # positive into the converter and therefore have the opposite sign.
        "P": -trace(problem, values, gfl_model, "P"),
        "Q": -trace(problem, values, gfl_model, "Q"),
        "Vdc_cap": trace(problem, values, gfl_model, "Vdc_cap"),
    }
    print(
        f"PF={pf.results.converged}, PF3={pf3.results.converged}, initialized={initialized}, "
        f"converged={converged}, "
        f"post_init_residual={(report.final_residual_inf if report is not None else float('nan')):.6e}, "
        f"steps={len(time)-1}"
    )
    for name, signal in signals.items():
        print(
            f"{name}: {signal[0]:.9g}->{signal[-1]:.9g}, "
            f"endpoint_delta={signal[-1]-signal[0]:+.6e}, "
            f"cycle_mean_delta={cycle_mean_drift(time, signal):+.6e}, "
            f"peak_to_peak={np.ptp(signal):.6e}"
        )
    if os.environ.get("VERAGRID_GFL_2BUS_DIAGNOSTICS", "0") == "1":
        helper.print_initialization_diagnostics(
            problem, gfl_model, time, values, np.asarray(_derivatives), x0_seed, dx0_seed
        )

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    labels = (("omega", "omega (p.u.)"), ("P", "P (p.u.)"), ("Q", "Q (p.u.)"), ("Vdc_cap", "Vdc (p.u.)"))
    for axis, (name, ylabel) in zip(axes, labels):
        axis.plot(1e3 * time, signals[name], linewidth=1.0)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (ms)")
    axes[0].set_title(f"Two-bus direct-generator GFL test ({SLACK_SOURCE} slack)")
    if P_EVENT_TIME >= 0.0:
        for axis in axes:
            axis.axvline(1e3 * P_EVENT_TIME, color="black", linestyle="--", linewidth=0.8)
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "event" if P_EVENT_TIME >= 0.0 else "steady_state"
    output = PLOT_OUTPUT_DIR / f"gfl_two_bus_{suffix}_{SLACK_SOURCE}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"plot={output}")


if __name__ == "__main__":
    main()
