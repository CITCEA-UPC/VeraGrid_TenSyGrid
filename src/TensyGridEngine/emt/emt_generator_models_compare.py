#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import VeraGridEngine.api as vg
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.EMT.solvers.jit_symbolic_solver import JitSymbolicSolver
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Templates.Emt.simple_generator_emt_template import get_simple_generator_emt_template as get_simple_generator_reference_template
from VeraGridEngine.Templates.Emt.simple_generator_emt_trig_template import get_simple_generator_emt_template_trig_transform


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
    opts = vg.EmtOptions(
        time_step=5e-6,
        simulation_time=0.04,
        tolerance=1e-6,
        solver_type=vg.EmtSolverTypes.Symbolic,
        integration_method=vg.DynamicIntegrationMethod.DaeTrapezoidal,
        verbose=1,
    )
    if not hasattr(opts, "newton_max_iter"):
        opts.newton_max_iter = 20
    return opts


def _find_var_by_prefix(results: vg.EmtResults, device: object, prefix: str) -> np.ndarray:
    vars_ = results.devices_vars_info[device]
    var = next(v for v in vars_ if v.name.startswith(prefix))
    idx = results.uid2idx_vars[var.uid]
    return results.values[:, idx, 0]


def _find_y_by_prefix(vars_by_uid, uid2idx, y: np.ndarray, prefix: str) -> np.ndarray | None:
    for var in vars_by_uid:
        if var.name.startswith(prefix):
            return y[:, uid2idx[var.uid]]
    return None


def run_case(generator_builder, case_name: str) -> dict[str, np.ndarray]:
    grid = vg.MultiCircuit(Sbase=2.0, fbase=50.0)
    vnom = 10.0

    bus0 = vg.Bus(name="Bus0", Vnom=vnom, is_slack=True)
    bus1 = vg.Bus(name="Bus1", Vnom=vnom)
    grid.add_bus(bus0)
    grid.add_bus(bus1)

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
    p_ph = (v_ph ** 2) / r_ph
    load = vg.Load(name="load", P1=p_ph, P2=p_ph, P3=p_ph, Q1=0.0, Q2=0.0, Q3=0.0)
    load.conn = vg.ShuntConnectionType.GroundedStar

    gen = vg.Generator(name="Gen0", vset=1.0, Snom=grid.Sbase, freq=50.0, r1=0.001, x1=1.7)

    grid.add_line(line)
    grid.add_generator(bus=bus0, api_obj=gen)
    grid.add_load(bus=bus1, api_obj=load)

    for bus in grid.buses:
        get_bus_emt_template(grid, bus)

    line_mdl = vg.get_pi_line_emt_template(vf=grid.var_factory, phN=False, phA=True, phB=True, phC=True).block
    load_mdl = vg.get_shunt_r_emt_template(vf=grid.var_factory, phA=True, phB=True, phC=True).block
    gen_mdl = generator_builder(vf=grid.var_factory).block

    vg.set_emt_model(device=line, model=line_mdl, var_factory=grid.var_factory)
    vg.set_emt_model(device=load, model=load_mdl, var_factory=grid.var_factory)
    vg.set_emt_model(device=gen, model=gen_mdl, var_factory=grid.var_factory)

    grid.add_emt_events_group(vg.EmtEventsGroup(name=f"{case_name}_group"))

    pf_driver = vg.PowerFlowDriver3Ph(grid=grid, options=build_pf_options())
    pf_driver.run()

    if not pf_driver.results.converged:
        raise RuntimeError(f"Power flow did not converge for {case_name}")

    options = build_emt_options()
    problem = EmtProblemDae(grid=grid, options=options, pf_results_3ph=pf_driver.results, pf_results=None)
    solver = JitSymbolicSolver(
        problem=problem,
        t0=0.0,
        t_end=options.simulation_time,
        h=options.time_step,
        method=options.integration_method,
        pred_method=vg.DynamicIntegrationMethod.OdeEuler,
        dense_threshold=0,
        verbose=False,
    )
    t, y, _, _, _ = solver.simulate(boundary_updater=problem)

    vars_by_uid = problem.get_device_vars_dict()[gen]
    bus_vars_by_uid = problem.get_device_vars_dict()[bus1]
    uid2idx = problem.uid2idx_vars

    i_a_var = next(v for v in vars_by_uid if v.name.startswith("i_A_"))
    omega_var = next(v for v in vars_by_uid if v.name.startswith("omega_"))
    va_var = next(v for v in bus_vars_by_uid if v.name.startswith("v_A_"))

    i_a = y[:, uid2idx[i_a_var.uid]]
    omega = y[:, uid2idx[omega_var.uid]]
    bus1_va = y[:, uid2idx[va_var.uid]]
    theta = _find_y_by_prefix(vars_by_uid, uid2idx, y, "theta_")
    u_cos = _find_y_by_prefix(vars_by_uid, uid2idx, y, "u_cos")

    return {
        "t": np.asarray(t, dtype=float),
        "i_a": i_a,
        "omega": omega,
        "v_a_bus1": bus1_va,
        "theta": theta,
        "u_cos": u_cos,
    }


def main() -> None:
    show_plots = os.environ.get("VERAGRID_SHOW_PLOTS", "0").strip().lower() in ("1", "true", "yes")

    print("Running classic simple generator EMT model...")
    classic = run_case(get_simple_generator_reference_template, "classic_simple")

    print("Running trig-transform simple generator EMT model...")
    trig = run_case(get_simple_generator_emt_template_trig_transform, "trig_simple")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(classic["t"], classic["i_a"], label="Classic", linewidth=1.4)
    axes[0].plot(trig["t"], trig["i_a"], label="Trig transform", linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("Gen i_A [p.u.]")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best")

    axes[1].plot(classic["t"], classic["omega"], linewidth=1.4)
    axes[1].plot(trig["t"], trig["omega"], linewidth=1.2, linestyle="--")
    axes[1].set_ylabel("Gen omega [p.u.]")
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(classic["t"], classic["v_a_bus1"], linewidth=1.4)
    axes[2].plot(trig["t"], trig["v_a_bus1"], linewidth=1.2, linestyle="--")
    axes[2].set_ylabel("Bus1 v_A [p.u.]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.35)

    axes[0].set_title("EMT comparison: classic vs trig-transform simple generator")
    fig.tight_layout()

    if trig["theta"] is not None and trig["u_cos"] is not None:
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.plot(trig["t"], np.cos(trig["theta"]), label="cos(theta) from EMT state", linewidth=1.4)
        ax2.plot(trig["t"], trig["u_cos"], label="u_cos state", linewidth=1.2, linestyle="--")
        ax2.set_title("Multilinear generator check: cos(theta) vs u_cos")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Value [p.u.]")
        ax2.grid(True, alpha=0.35)
        ax2.legend(loc="best")
        fig2.tight_layout()
        mismatch = np.asarray(trig["u_cos"]) - np.cos(np.asarray(trig["theta"]))
        rmse_ucos = float(np.sqrt(np.mean(mismatch ** 2)))
        max_ucos = float(np.max(np.abs(mismatch)))
        print(f"RMSE(u_cos - cos(theta)): {rmse_ucos:.6e}")
        print(f"MAX |u_cos - cos(theta)|: {max_ucos:.6e}")

    diff_i = classic["i_a"] - trig["i_a"]
    rmse_i = float(np.sqrt(np.mean(diff_i ** 2)))
    max_i = float(np.max(np.abs(diff_i)))
    print(f"RMSE(Delta i_A): {rmse_i:.6e}")
    print(f"MAX |Delta i_A|: {max_i:.6e}")

    plt.show()


if __name__ == "__main__":
    main()
