# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Load IEEE9 from Grids_and_profiles, attach EMT models, and simulate it."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
for import_path in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import VeraGridEngine.api as gce
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.emt_solver_factory import build_emt_solver
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowDriver
from VeraGridEngine.Simulations.PowerFlow3ph.power_flow_driver_3ph import PowerFlowDriver3Ph
from VeraGridEngine.Templates.Emt.load_RLC_emt_template import get_shunt_r_emt_template
from VeraGridEngine.Templates.Emt.load_zip_emt_template import get_load_ZIP_emt_template
from VeraGridEngine.Templates.Emt.generator_emt_type_template import get_complete_generator_template_emt
from VeraGridEngine.Templates.Emt.pi_line_emt_template import get_pi_line_emt_template
from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import (
    get_generator_thevenin_rl_emt_template_with_ref,
)
from VeraGridEngine.Templates.Emt.transformer_emt_template import (
    get_series_transformer_emt_template,
    get_transformer_emt_template,
)
from VeraGridEngine.Utils.Symbolic.block import find_name_in_block
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtSolverTypes,
)

from ieee9_emt_simulation import build_power_flow_options

USE_CONVENTIONAL_THREE_PHASE_BASE = True
USE_STABLE_FROZEN_EXCITATION = True
USE_COUPLED_SETTLING_INITIALIZATION = False
GENERATOR_MECHANICAL_DAMPING = 0.0
COUPLED_SETTLING_TIME = 8.0
COUPLED_SETTLING_TIME_STEP = 100.0e-6
IEEE9_GRID_PATH = REPO_ROOT / "Grids_and_profiles" / "grids" / "IEEE 9 Bus.gridcal"


def _set_constant_power_zip_coefficients(block: Any) -> None:
    """Configure the EMT ZIP load as pure constant P/Q."""
    for coefficient, value in (
        ("a1", 0.0), ("a2", 0.0), ("a3", 1.0),
        ("a4", 0.0), ("a5", 0.0), ("a6", 1.0),
    ):
        block.set_parameter_in_model(var_name=coefficient, new_value=value)


def build_ieee9_multicircuit(
    replace_bus2_generator_with_termination: bool = True,
) -> gce.MultiCircuit:
    """Load IEEE9 from ``Grids_and_profiles`` and apply the staged EMT setup."""
    if not IEEE9_GRID_PATH.is_file():
        raise FileNotFoundError(f"IEEE9 grid file not found: {IEEE9_GRID_PATH}")
    grid = gce.open_file(str(IEEE9_GRID_PATH))
    if len(grid.buses) != 9 or len(grid.generators) != 3:
        raise RuntimeError(
            f"Unexpected IEEE9 contents: {len(grid.buses)} buses, "
            f"{len(grid.generators)} generators"
        )

    buses_by_name = {bus.name: bus for bus in grid.buses}
    generator_bus_names = {generator.bus.name for generator in grid.generators}

    # The canonical file stores the three unity-ratio generator step-up
    # branches as zero-R, zero-B lines. Convert those device objects in memory;
    # MultiCircuit preserves their status/rating profiles during conversion.
    step_up_lines = [
        line for line in list(grid.lines)
        if (
            (line.bus_from.name in generator_bus_names or line.bus_to.name in generator_bus_names)
            and abs(float(line.R)) < 1.0e-12
            and abs(float(line.B)) < 1.0e-12
        )
    ]
    if len(step_up_lines) != 3:
        raise RuntimeError(f"Expected three IEEE9 generator step-up branches, found {len(step_up_lines)}")
    for idx, line in enumerate(step_up_lines):
        original_name = line.name
        transformer = grid.convert_line_to_transformer(line)
        transformer.name = f"transformer {idx} ({original_name})"
        transformer.HV = transformer.bus_to.Vnom
        transformer.LV = transformer.bus_from.Vnom

    # Stage 1 of the one-by-one replacement keeps the canonical slack and
    # bus-1 generators, while bus 2 receives the temporary resistive ending.
    if replace_bus2_generator_with_termination:
        generator_bus_2 = next(gen for gen in grid.generators if gen.bus.name == "bus 2")
        grid.delete_generator(generator_bus_2)
        grid.add_load(
            bus=buses_by_name["bus 2"],
            api_obj=gce.Load(name="temporary termination bus 2", P=1.0, Q=0.0),
        )

    # Machine electrical parameters are not populated in this static IEEE9
    # file. Supply only the parameters required by the attached EMT models.
    for generator in grid.generators:
        generator.Snom = grid.Sbase
        if generator.bus.is_slack:
            generator.R1 = 0.002
            generator.X1 = 0.25
        else:
            generator.R1 = 0.001
            generator.X1 = 1.7

    return grid


def build_emt_options() -> EmtOptions:
    return EmtOptions(
        time_step=5.0e-6,
        simulation_time=0.1,
        tolerance=1.0e-6,
        solver_type=EmtSolverTypes.Symbolic,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        # The ring-closed IEEE9 reduced initializer currently crashes inside
        # SciPy/SuperLU's sparse spsolve path. The dense reduced system is small
        # and converges to below 1e-8; time stepping remains symbolic/sparse.
        init_dense_threshold=10_000,
        conventional_three_phase_base=USE_CONVENTIONAL_THREE_PHASE_BASE,
        verbose=0,
    )


def attach_baseline_emt_models(
        grid: gce.MultiCircuit,
        frozen_generator_controls: bool = False,
        frozen_generator_excitation: bool = False,
        generator_mechanical_damping: float = GENERATOR_MECHANICAL_DAMPING,
        frozen_generator_e_qp: bool = False,
        use_thevenin_for_dynamic_generators: bool = False,
        use_constant_power_loads: bool = True,
        dynamic_generator_builder: Any | None = None,
) -> None:
    """Attach the staged source/generator, PI-line, transformer, and load models."""
    for bus in grid.buses:
        get_bus_emt_template(grid=grid, bus=bus)

    for idx, generator in enumerate(grid.generators):
        if generator.bus.is_slack or use_thevenin_for_dynamic_generators:
            model = get_generator_thevenin_rl_emt_template_with_ref(
                vf=grid.var_factory,
                name="emt_thevenin_slack",
            ).block
        else:
            if dynamic_generator_builder is None:
                # Use the same controller-complete Sauer-Pai generator as the
                # validated generator + PI-line + R-load EMT example.
                model = get_complete_generator_template_emt(
                    vf=grid.var_factory,
                    name=f"emt_complete_generator_{idx}",
                    conventional_three_phase_base=USE_CONVENTIONAL_THREE_PHASE_BASE,
                    mechanical_damping=generator_mechanical_damping,
                    frozen_controls=frozen_generator_controls,
                    frozen_excitation=frozen_generator_excitation,
                    freeze_e_qp=frozen_generator_e_qp,
                ).block
            else:
                model = dynamic_generator_builder(
                    grid.var_factory,
                    f"emt_multilinear_generator_{idx}",
                )
        set_emt_model(device=generator, model=model, var_factory=grid.var_factory)

    for line in grid.lines:
        model = get_pi_line_emt_template(
            vf=grid.var_factory,
            phN=False,
            phA=True,
            phB=True,
            phC=True,
            name=line.name,
        ).block
        set_emt_model(device=line, model=model, var_factory=grid.var_factory)

    for transformer in grid.transformers2w:
        if (
            abs(float(transformer.G)) <= 1.0e-12
            and abs(float(transformer.B)) <= 1.0e-12
            and abs(float(transformer.tap_phase)) <= 1.0e-12
        ):
            model = get_series_transformer_emt_template(
                vf=grid.var_factory,
                name=transformer.name,
            ).block
        else:
            model = get_transformer_emt_template(
                vf=grid.var_factory,
                name=transformer.name,
            ).block
        set_emt_model(device=transformer, model=model, var_factory=grid.var_factory)

    for idx, load in enumerate(grid.loads, start=1):
        if load.name.startswith("temporary termination"):
            model = get_shunt_r_emt_template(
                vf=grid.var_factory,
                phA=True,
                phB=True,
                phC=True,
                name=f"resistive_load_{idx}_{load.name}",
                conventional_three_phase_base=USE_CONVENTIONAL_THREE_PHASE_BASE,
            ).block
        else:
            model = get_load_ZIP_emt_template(
                vf=grid.var_factory,
                phA=True,
                phB=True,
                phC=True,
                connection_type=None,
                name=f"constant_pq_load_{idx}_{load.name}",
                conventional_three_phase_base=USE_CONVENTIONAL_THREE_PHASE_BASE,
            ).block
            if use_constant_power_loads:
                _set_constant_power_zip_coefficients(model)
        set_emt_model(device=load, model=model, var_factory=grid.var_factory)


def apply_emt_snapshot_as_initial_condition(
        problem: EmtProblemDae,
        values: np.ndarray,
        differential_values: np.ndarray,
) -> None:
    """Replace the problem's initial point by one complete DAE snapshot.

    A phase-aligned snapshot contains the coupled network, machine and controller
    state that device-local PF initialization cannot construct.  Both algebraic
    values and differential values are retained so the restarted trapezoidal
    solve begins on the same periodic trajectory instead of projecting onto it.
    """
    all_vars = problem.get_state_vars() + problem.get_algebraic_vars()
    diff_vars = problem.get_diff_vars()
    if values.shape[0] != len(all_vars):
        raise ValueError(f"Expected {len(all_vars)} EMT values, got {values.shape[0]}")
    if differential_values.shape[0] != len(diff_vars):
        raise ValueError(
            f"Expected {len(diff_vars)} EMT differential values, "
            f"got {differential_values.shape[0]}"
        )

    for var, value in zip(all_vars, values):
        problem.init_guess[var.uid] = float(value)
    for diff_var, value in zip(diff_vars, differential_values):
        problem.diff_init_guess[diff_var.uid] = float(value)


def settle_and_reseed_emt_problem(
        problem: EmtProblemDae,
        options: EmtOptions,
        settling_time: float,
        settling_time_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Settle the coupled DAE and reuse its phase-aligned endpoint as ``t=0``.

    ``settling_time`` must contain an integer number of fundamental cycles so
    resetting simulation time does not rotate the network waveform relative to
    stored inductor, capacitor and rotor states.
    """
    cycles = settling_time * problem.grid.fBase
    if abs(cycles - round(cycles)) > 1.0e-9:
        raise ValueError("settling_time must be an integer number of fundamental cycles")

    settling_solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=settling_time,
        h=settling_time_step,
        method=options.integration_method,
    )
    time, values, differential_values, initialized, converged = settling_solver.simulate(
        boundary_updater=problem
    )
    if not initialized or not converged or not np.isfinite(values[-1]).all():
        raise RuntimeError("Coupled EMT settling run did not produce a valid endpoint")

    apply_emt_snapshot_as_initial_condition(
        problem=problem,
        values=values[-1],
        differential_values=differential_values[-1],
    )
    problem.step_counter = 0
    return time, values, differential_values, initialized, converged


def run_simulation() -> tuple[EmtProblemDae, np.ndarray, np.ndarray, bool, bool]:
    grid = build_ieee9_multicircuit()

    # Stable reference stage: Thevenin sources, resistive loads, and PI lines.
    attach_baseline_emt_models(
        grid,
        frozen_generator_excitation=USE_STABLE_FROZEN_EXCITATION,
    )

    power_flow = PowerFlowDriver(grid=grid, options=build_power_flow_options())
    power_flow.run()
    if not bool(power_flow.results.converged):
        raise RuntimeError("The IEEE 9-bus balanced power flow did not converge")

    power_flow_3ph = PowerFlowDriver3Ph(grid=grid, options=build_power_flow_options())
    power_flow_3ph.run()
    if not bool(power_flow_3ph.results.converged):
        raise RuntimeError("The IEEE 9-bus three-phase power flow did not converge")

    options = build_emt_options()
    problem = EmtProblemDae(
        grid=grid,
        options=options,
        pf_results_3ph=power_flow_3ph.results,
        pf_results=power_flow.results,
    )
    if USE_COUPLED_SETTLING_INITIALIZATION:
        settle_and_reseed_emt_problem(
            problem=problem,
            options=options,
            settling_time=COUPLED_SETTLING_TIME,
            settling_time_step=COUPLED_SETTLING_TIME_STEP,
        )
    solver = build_emt_solver(
        options=options,
        problem=problem,
        t0=0.0,
        t_end=float(options.simulation_time),
        h=float(options.time_step),
        method=options.integration_method,
    )
    t_arr, y_arr, _dy_arr, initialized, converged = solver.simulate(
        boundary_updater=cast(Any, problem)
    )
    return problem, t_arr, y_arr, bool(initialized), bool(converged)


def save_plots(problem: EmtProblemDae, t_arr: np.ndarray, y_arr: np.ndarray) -> list[Path]:
    """Save network-wide voltage, load-bus voltage, and source-current plots."""
    output_dir = Path(__file__).parent
    time_ms = t_arr * 1e3
    output_paths: list[Path] = []

    all_bus_path = output_dir / "ieee9_emt_all_bus_voltage_a.png"
    fig, axis = plt.subplots(figsize=(10.0, 5.5), constrained_layout=True)
    for bus in problem.grid.buses:
        voltage_idx = int(problem.get_var_idx(bus.emt_model.out_vars[0]))
        axis.plot(time_ms, y_arr[:, voltage_idx], linewidth=0.9, label=bus.name)
    axis.set_title("IEEE9 EMT phase-A voltage at all buses")
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Voltage (p.u.)")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=3, fontsize=8)
    fig.savefig(all_bus_path, dpi=180)
    plt.close(fig)
    output_paths.append(all_bus_path)

    real_load_buses = [load.bus for load in problem.grid.loads if load.name.startswith("load ")]
    load_path = output_dir / "ieee9_emt_load_bus_voltages.png"
    fig, axes = plt.subplots(len(real_load_buses), 1, figsize=(10.0, 8.0), sharex=True, constrained_layout=True)
    for axis, bus in zip(np.atleast_1d(axes), real_load_buses):
        for phase_idx, phase in enumerate(("A", "B", "C")):
            voltage_idx = int(problem.get_var_idx(bus.emt_model.out_vars[phase_idx]))
            axis.plot(time_ms, y_arr[:, voltage_idx], linewidth=0.9, label=f"phase {phase}")
        axis.set_title(bus.name)
        axis.set_ylabel("Voltage (p.u.)")
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("IEEE9 EMT three-phase load-bus voltages")
    fig.savefig(load_path, dpi=180)
    plt.close(fig)
    output_paths.append(load_path)

    source_path = output_dir / "ieee9_emt_source_currents.png"
    source_model = problem.grid.generators[0].emt_model
    fig, axis = plt.subplots(figsize=(10.0, 4.5), constrained_layout=True)
    for phase in ("A", "B", "C"):
        current_var = find_name_in_block(f"i_{phase}", source_model)
        if current_var is None:
            raise RuntimeError(f"Could not find source current i_{phase}")
        current_idx = int(problem.get_var_idx(current_var))
        axis.plot(time_ms, y_arr[:, current_idx], linewidth=0.9, label=f"i{phase}")
    axis.set_title("IEEE9 EMT Thevenin-source currents")
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel("Current (p.u.)")
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=3)
    fig.savefig(source_path, dpi=180)
    plt.close(fig)
    output_paths.append(source_path)

    for generator in problem.grid.generators:
        if generator.bus.is_slack:
            continue
        omega_var = next(
            (variable for variable in generator.emt_model.state_vars if variable.name.startswith("omega")),
            None,
        )
        theta_var = next(
            (variable for variable in generator.emt_model.state_vars if variable.name.startswith("theta_")),
            None,
        )
        if omega_var is None or theta_var is None:
            raise RuntimeError(f"Could not find rotor states for {generator.name}")
        state_path = output_dir / f"ieee9_emt_{generator.name.replace(' ', '_')}_rotor_states.png"
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), sharex=True, constrained_layout=True)
        axes[0].plot(time_ms, y_arr[:, int(problem.get_var_idx(omega_var))])
        axes[0].set_ylabel("Speed (p.u.)")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(time_ms, y_arr[:, int(problem.get_var_idx(theta_var))])
        axes[1].set_ylabel("Angle (rad)")
        axes[1].set_xlabel("Time (ms)")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(f"IEEE9 EMT rotor states — {generator.name}")
        fig.savefig(state_path, dpi=180)
        plt.close(fig)
        output_paths.append(state_path)
    return output_paths


def main() -> None:
    problem, t_arr, y_arr, initialized, converged = run_simulation()
    load_buses = [load.bus for load in problem.grid.loads]
    plot_paths = save_plots(problem=problem, t_arr=t_arr, y_arr=y_arr)

    print("IEEE 9-bus Thevenin-source EMT baseline")
    print(
        f"buses={len(problem.grid.buses)}, generators={len(problem.grid.generators)}, "
        f"loads={len(problem.grid.loads)}, lines={len(problem.grid.lines)}, "
        f"transformers={len(problem.grid.transformers2w)}"
    )
    print(f"initialized={initialized}, converged={converged}, steps={len(t_arr) - 1}")
    print("plots=" + ",".join(str(path) for path in plot_paths))
    print("time_s," + ",".join(f"vA_{bus.name}" for bus in load_buses))
    sample_steps = np.linspace(0, len(t_arr) - 1, min(21, len(t_arr)), dtype=int)
    for step in sample_steps:
        voltages = [
            y_arr[step, int(problem.get_var_idx(bus.emt_model.out_vars[0]))]
            for bus in load_buses
        ]
        print(f"{t_arr[step]:.6e}," + ",".join(f"{value:.8e}" for value in voltages))


if __name__ == "__main__":
    main()
