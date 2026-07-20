#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import VeraGridEngine.api as vg


def find_var_in_block(block: vg.Block, name: str):
    for var in block.algebraic_vars + block.state_vars + list(block.event_dict.keys()) + block.diff_vars:
        if var.name == name:
            return var

    for child in block.children:
        found = find_var_in_block(child, name)
        if found is not None:
            return found

    return None


def build_pf_options() -> vg.PowerFlowOptions:
    return vg.PowerFlowOptions(
        solver_type=vg.SolverType.NR,
        retry_with_other_methods=False,
        verbose=0,
        initialize_with_existing_solution=True,
        tolerance=1e-6,
        max_iter=25,
        control_q=False,
        control_taps_modules=True,
        control_taps_phase=True,
        control_remote_voltage=True,
        orthogonalize_controls=True,
        apply_temperature_correction=True,
        branch_impedance_tolerance_mode=vg.BranchImpedanceMode.Specified,
        distributed_slack=False,
        ignore_single_node_islands=False,
        trust_radius=1.0,
        backtracking_parameter=0.05,
        use_stored_guess=False,
        initialize_angles=False,
        generate_report=False,
    )


def build_emt_options() -> vg.EmtOptions:
    options = vg.EmtOptions(
        time_step=5e-6,
        simulation_time=0.06,
        tolerance=1e-6,
        solver_type=vg.EmtSolverTypes.Symbolic,
        integration_method=vg.DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=vg.EmtInitializationMethod.Auto,
        verbose=0,
    )

    if not hasattr(options, "newton_max_iter"):
        options.newton_max_iter = 20

    return options


def extract_bus_voltages(results: vg.EmtResults, bus: vg.Bus) -> dict[str, np.ndarray]:
    bus_vars = results.devices_vars_info[bus]
    out: dict[str, np.ndarray] = {}

    for phase in ("A", "B", "C"):
        var = next(v for v in bus_vars if v.name.startswith(f"v_{phase}_"))
        idx = results.uid2idx_vars[var.uid]
        out[phase] = results.values[:, idx, 0]

    return out


def run_case(line_model: str) -> dict[str, np.ndarray]:
    if line_model not in ("pi", "bergeron"):
        raise ValueError("line_model must be 'pi' or 'bergeron'")

    grid = vg.MultiCircuit(Sbase=2.0, fbase=50.0)

    vnom = 10.0
    bus0 = vg.Bus(name="Bus0", Vnom=vnom, is_slack=True)
    bus1 = vg.Bus(name="Bus1", Vnom=vnom)
    grid.add_bus(bus0)
    grid.add_bus(bus1)

    for bus in grid.buses:
        bus.emt_model = vg.BusEmtTemplate(
            vf=grid.var_factory,
            mask=[False, True, True, True],
            is_dc=bus.is_dc,
            name=f"{bus.name}_emt_template",
        ).block

    line = vg.Line(name="line0", bus_from=bus0, bus_to=bus1, length=10.0, rate=900.0)
    tower = vg.OverheadLineType(name="Tower", Vnom=vnom)
    wire = vg.Wire(
        name="Panther 30/7 ACSR",
        diameter=21.0,
        diameter_internal=9.0,
        is_tube=True,
        r=0.1363,
        max_current=1,
    )
    tower.add_wire_relationship(wire=wire, xpos=-12.65, ypos=27.5, phase=1)
    tower.add_wire_relationship(wire=wire, xpos=0.0, ypos=27.5, phase=2)
    tower.add_wire_relationship(wire=wire, xpos=12.65, ypos=27.5, phase=3)
    tower.compute()
    line.apply_template(tower, grid.Sbase, grid.fBase)

    r_ph = 100.0
    v_ph = vnom / np.sqrt(3.0)
    p_ph = (v_ph**2) / r_ph
    load = vg.Load(name="load", P1=p_ph, P2=p_ph, P3=p_ph, Q1=0.0, Q2=0.0, Q3=0.0)
    load.conn = vg.ShuntConnectionType.GroundedStar

    gen = vg.Generator(name="Gen0", vset=1.0, Snom=grid.Sbase, freq=50.0, r1=0.001, x1=1.7)

    grid.add_line(line)
    grid.add_generator(bus=bus0, api_obj=gen)
    grid.add_load(bus=bus1, api_obj=load)

    gen_mdl = vg.get_generator_thevenin_rl_emt_template_with_ref(vf=grid.var_factory).block
    if line_model == "pi":
        line_mdl = vg.get_pi_line_emt_template(vf=grid.var_factory, phN=False, phA=True, phB=True, phC=True).block
    else:
        line_mdl = vg.get_bergeron_line_emt_template(vf=grid.var_factory, phN=False, phA=True, phB=True, phC=True).block

    load_mdl = vg.get_shunt_r_emt_template(vf=grid.var_factory, phA=True, phB=True, phC=True).block

    vg.set_emt_model(device=gen, model=gen_mdl, var_factory=grid.var_factory)
    vg.set_emt_model(device=line, model=line_mdl, var_factory=grid.var_factory)
    vg.set_emt_model(device=load, model=load_mdl, var_factory=grid.var_factory)

    events_group = vg.EmtEventsGroup(name=f"{line_model}_line_group")
    grid.add_emt_events_group(events_group)

    r_a = find_var_in_block(load_mdl, "R_A_Shunt_R_3ph")
    r_b = find_var_in_block(load_mdl, "R_B_Shunt_R_3ph")
    r_c = find_var_in_block(load_mdl, "R_C_Shunt_R_3ph")
    for parameter in (r_a, r_b, r_c):
        grid.add_emt_event(vg.EmtEvent(device=load, parameter=parameter, time=0.02, value=3.0, group=events_group))
        grid.add_emt_event(vg.EmtEvent(device=load, parameter=parameter, time=0.04, value=6.0, group=events_group))

    pf_driver = vg.PowerFlowDriver3Ph(grid=grid, options=build_pf_options())
    pf_driver.run()

    if not pf_driver.results.converged:
        raise RuntimeError(f"Power flow did not converge for {line_model} case")

    emt_driver = vg.EmtSimulationDriver(
        grid=grid,
        options=build_emt_options(),
        pf_results_3ph=pf_driver.results,
    )
    emt_driver.run()

    if emt_driver.results is None:
        raise RuntimeError(f"EMT results are missing for {line_model} case")

    results = emt_driver.results
    time_s = (results.time_array.asi8 - results.time_array.asi8[0]) * 1e-9
    voltages = extract_bus_voltages(results=results, bus=bus1)

    return {
        "t": time_s,
        "v_a": voltages["A"],
        "v_b": voltages["B"],
        "v_c": voltages["C"],
    }


def main() -> None:
    show_plots = os.environ.get("VERAGRID_SHOW_PLOTS", "0").strip() in ("1", "true", "True")

    print("Running PI-line EMT case...")
    pi_case = run_case("pi")

    print("Running Bergeron-line EMT case...")
    bergeron_case = run_case("bergeron")

    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    for i, phase in enumerate(("a", "b", "c")):
        axes[i].plot(pi_case["t"], pi_case[f"v_{phase}"], label="PI line", linewidth=1.4)
        axes[i].plot(bergeron_case["t"], bergeron_case[f"v_{phase}"], label="Bergeron line", linewidth=1.2, linestyle="--")
        axes[i].set_ylabel(f"V{phase.upper()} [p.u.]")
        axes[i].grid(True, alpha=0.35)

    axes[0].set_title("EMT comparison at Bus1: PI vs Bergeron line")
    axes[2].set_xlabel("Time [s]")
    axes[0].legend(loc="best")

    fig.tight_layout()
    waveform_path = output_dir / "emt_line_type_comparison_bus1_voltages.png"
    fig.savefig(waveform_path, dpi=160)

    diff_a = pi_case["v_a"] - bergeron_case["v_a"]
    rmse = float(np.sqrt(np.mean(diff_a**2)))
    max_abs = float(np.max(np.abs(diff_a)))

    fig2, ax2 = plt.subplots(figsize=(11, 3.8))
    ax2.plot(pi_case["t"], diff_a, color="tab:red", linewidth=1.2)
    ax2.axhline(0.0, color="black", linewidth=0.9)
    ax2.grid(True, alpha=0.35)
    ax2.set_title("Difference (PI - Bergeron) on Bus1 phase-A voltage")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Delta V_A [p.u.]")
    fig2.tight_layout()

    diff_path = output_dir / "emt_line_type_difference_phase_a.png"
    fig2.savefig(diff_path, dpi=160)

    print(f"Saved: {waveform_path}")
    print(f"Saved: {diff_path}")
    print(f"RMSE(Delta V_A): {rmse:.6e}")
    print(f"MAX |Delta V_A|: {max_abs:.6e}")

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
