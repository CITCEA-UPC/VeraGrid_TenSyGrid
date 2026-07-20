#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import VeraGridEngine.api as gce
from VeraGridEngine.enumerations import (
    DynamicIntegrationMethod,
    EmtInitializationMethod,
    EmtSolverTypes,
    SolverType,
    WindingType,
)
from VeraGridEngine.Simulations.PowerFlow.power_flow_driver import PowerFlowOptions
from VeraGridEngine.Simulations.EMT.emt_options import EmtOptions
from VeraGridEngine.Simulations.EMT.problems.emt_problem_dae import EmtProblemDae
from VeraGridEngine.Simulations.EMT.solvers.jit_symbolic_solver import JitSymbolicSolver
from VeraGridEngine.Utils.Symbolic.bus_emt_template import get_bus_emt_template
from VeraGridEngine.Templates.Emt.load_RLC_emt_template import get_shunt_r_emt_template
from VeraGridEngine.Templates.Emt.thevenin_equivalent_emt_generator_template import (
    get_generator_thevenin_rl_emt_template_with_ref,
)
from VeraGridEngine.Templates.Emt.xfmr_emt_template import get_xfmr_emt_template
from VeraGridEngine.Templates.Emt.xfmr_emt_multilinear_template import get_xfmr_emt_template_multilinear
from VeraGridEngine.Utils.Symbolic.templates_common_functions import set_emt_model
from VeraGridEngine.Utils.Symbolic.block import Block


def find_name_in_block(name: str, block: Block):
    for var in block.algebraic_vars + block.state_vars + list(block.event_dict.keys()) + block.diff_vars:
        if var.name == name:
            return var
    for child in block.children:
        res = find_name_in_block(name, child)
        if res is not None:
            return res
    return None


def get_series(problem: EmtProblemDae, y: np.ndarray, var):
    if var is None:
        return None
    return y[:, problem.get_var_idx(var)]


def build_grid() -> tuple[gce.MultiCircuit, object, object, object, object, object]:
    circuit = gce.MultiCircuit(name="XFMR EMT compare", Sbase=100.0, fbase=50.0)

    bus_hv = gce.Bus(name="Bus_HV", Vnom=230.0, is_slack=True)
    bus_lv = gce.Bus(name="Bus_LV", Vnom=66.0)
    circuit.add_bus(bus_hv)
    circuit.add_bus(bus_lv)

    gen = gce.Generator(name="Generator", vset=1.0, Snom=200.0, freq=50.0, r1=0.002, x1=0.25)
    circuit.add_generator(bus_hv, gen)

    xfmr = gce.Transformer2W(
        name="Transformer",
        bus_from=bus_hv,
        bus_to=bus_lv,
        HV=230.0,
        LV=66.0,
        nominal_power=100.0,
        copper_losses=120.0,
        iron_losses=35.0,
        no_load_current=0.20,
        short_circuit_voltage=10.0,
        rate=100.0,
        tap_module=1.0,
        tap_phase=0.0,
    )
    xfmr.conn_f = WindingType.GroundedStar
    xfmr.conn_t = WindingType.GroundedStar
    xfmr.vector_group_number = 0
    xfmr.fill_design_properties(Pcu=120.0, Pfe=35.0, I0=0.20, Vsc=10.0, Sbase=circuit.Sbase)
    circuit.add_transformer2w(xfmr)

    load = gce.Load(
        name="Load",
        P=40.0,
        Q=0.0,
        P1=40.0 / 3.0,
        P2=40.0 / 3.0,
        P3=40.0 / 3.0,
        Q1=0.0,
        Q2=0.0,
        Q3=0.0,
    )
    circuit.add_load(bus_lv, load)

    for bus in circuit.buses:
        get_bus_emt_template(grid=circuit, bus=bus)

    gen_mdl = get_generator_thevenin_rl_emt_template_with_ref(vf=circuit.var_factory, name="gen_test").block
    load_mdl = get_shunt_r_emt_template(vf=circuit.var_factory, phA=True, phB=True, phC=True).block

    set_emt_model(device=gen, model=gen_mdl, var_factory=circuit.var_factory)
    set_emt_model(device=load, model=load_mdl, var_factory=circuit.var_factory)

    return circuit, bus_hv, bus_lv, gen, xfmr, load


def run_case(use_multilinear: bool) -> dict[str, np.ndarray]:
    circuit, _, bus_lv, _, xfmr, _ = build_grid()

    xfmr_builder = get_xfmr_emt_template_multilinear if use_multilinear else get_xfmr_emt_template
    xfmr_mdl = xfmr_builder(vf=circuit.var_factory, name="xfmr_test").block
    set_emt_model(device=xfmr, model=xfmr_mdl, var_factory=circuit.var_factory)

    xfmr_mdl.set_parameter_in_model(var_name="xfmr_use_linear_core_xfmr_test", new_value=1.0)
    xfmr_mdl.set_parameter_in_model(var_name="xfmr_c_term_xfmr_test", new_value=0.0)
    xfmr_mdl.set_parameter_in_model(var_name="xfmr_core_topology_code_xfmr_test", new_value=3.0)
    xfmr_mdl.set_parameter_in_model(var_name="xfmr_yoke_area_rel_xfmr_test", new_value=1.0)
    xfmr_mdl.set_parameter_in_model(var_name="xfmr_yoke_length_rel_xfmr_test", new_value=1.0)
    xfmr_mdl.set_parameter_in_model(var_name="xfmr_outer_leg_area_rel_xfmr_test", new_value=1.0)
    xfmr_mdl.set_parameter_in_model(var_name="xfmr_outer_leg_length_rel_xfmr_test", new_value=1.0)
    xfmr_mdl.set_parameter_in_model(var_name="xfmr_core_knee_flux_mult_xfmr_test", new_value=1.05)
    xfmr_mdl.set_parameter_in_model(var_name="xfmr_core_knee_current_mult_xfmr_test", new_value=8.0)

    pf_options = PowerFlowOptions(
        solver_type=SolverType.NR,
        verbose=False,
        retry_with_other_methods=True,
        max_iter=50,
        tolerance=1e-8,
        control_q=False,
        control_taps_modules=False,
        control_taps_phase=False,
        distributed_slack=False,
    )
    pf_results = gce.power_flow(grid=circuit, options=pf_options)
    if not pf_results.converged:
        raise RuntimeError("Power flow did not converge")

    options = EmtOptions(
        time_step=10e-6,
        simulation_time=0.06,
        tolerance=1e-6,
        solver_type=EmtSolverTypes.Symbolic,
        integration_method=DynamicIntegrationMethod.DaeTrapezoidal,
        initialization_method=EmtInitializationMethod.Explicit,
        verbose=1,
    )
    problem = EmtProblemDae(grid=circuit, options=options, pf_results_3Ph=None, pf_results=pf_results)
    solver = JitSymbolicSolver(
        problem=problem,
        t0=0.0,
        t_end=options.simulation_time,
        h=options.time_step,
        method=options.integration_method,
        pred_method=DynamicIntegrationMethod.OdeEuler,
        dense_threshold=0,
        verbose=False,
    )
    solver.build_jit_kernel(options.integration_method)
    solver._build_jit_symbolic_hybrid(options.integration_method, use_sparse=True)
    if options.integration_method == DynamicIntegrationMethod.DaeTrapezoidal:
        solver.build_jit_kernel(DynamicIntegrationMethod.DaeBackEuler)
        solver._build_jit_symbolic_hybrid(DynamicIntegrationMethod.DaeBackEuler, use_sparse=True)

    t, y, _, _, _ = solver.simulate(boundary_updater=problem)

    vA_lv = get_series(problem, y, find_name_in_block("v_A", bus_lv.emt_model))
    iA_f = get_series(problem, y, find_name_in_block("if_A", xfmr_mdl))
    iA_t = get_series(problem, y, find_name_in_block("it_A", xfmr_mdl))

    return {
        "t": np.asarray(t).reshape(-1),
        "vA_lv": np.asarray(vA_lv).reshape(-1),
        "iA_f": np.asarray(iA_f).reshape(-1),
        "iA_t": np.asarray(iA_t).reshape(-1),
    }


def main() -> None:
    print("Running reference transformer EMT model...")
    ref = run_case(use_multilinear=False)

    print("Running multilinear transformer EMT model...")
    ml = run_case(use_multilinear=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(ref["t"], ref["vA_lv"], label="Reference", linewidth=1.4)
    axes[0].plot(ml["t"], ml["vA_lv"], label="Multilinear", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Bus LV v_A [p.u.]")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best")

    axes[1].plot(ref["t"], ref["iA_f"], linewidth=1.4)
    axes[1].plot(ml["t"], ml["iA_f"], linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("XFMR if_A [p.u.]")
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(ref["t"], ref["iA_t"], linewidth=1.4)
    axes[2].plot(ml["t"], ml["iA_t"], linestyle="--", linewidth=1.2)
    axes[2].set_ylabel("XFMR it_A [p.u.]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.35)

    axes[0].set_title("EMT comparison: reference vs multilinear transformer")
    fig.tight_layout()

    for key in ("vA_lv", "iA_f", "iA_t"):
        diff = ref[key] - ml[key]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        dmax = float(np.max(np.abs(diff)))
        print(f"RMSE({key}): {rmse:.6e}")
        print(f"MAX |Delta {key}|: {dmax:.6e}")

    plt.show()


if __name__ == "__main__":
    main()
